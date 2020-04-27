import argparse
from pathlib import Path
import csv
from tqdm import tqdm
from datetime import datetime

import pandas as pd
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.util import dump_metrics

from adat.models import get_model_by_name
from adat.dataset import IDENTITY_TOKEN
from adat.dataset_readers.classifier import ClassificationReader
from adat.dataset_readers.copynet import CopyNetReader
from adat.attackers import AttackerOutput, Cascada
from adat.utils import load_weights, get_args_from_path


parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=-1)
parser.add_argument('--csv_path', type=str, default='data/test.csv', required=True)
parser.add_argument('--results_path', type=str, default='results')

parser.add_argument('--copynet_path', type=str, default='experiments/copynet')
parser.add_argument('--classifier_copynet_path', type=str, default='experiments/classification_copynet')
parser.add_argument('--classifier_path', type=str, default='experiments/classification')
parser.add_argument('--levenshtein_path', type=str, default='experiments/deep_levenshtein')

parser.add_argument('--prob_diff_weight', type=float, default=1.0)
parser.add_argument('--levenshtein_weight', type=float, default=0.1)
parser.add_argument('--learning_rate', type=float, default=2.0)
parser.add_argument('--max_steps', type=int, default=30)
parser.add_argument('--num_updates', type=int, default=1)
parser.add_argument('--beam_size', type=int, default=1)
parser.add_argument('--maskers', type=str, default=IDENTITY_TOKEN, help='string with comma-separated values')
parser.add_argument('--early_stopping', action='store_true')
parser.add_argument('--sample', type=int, default=None)


if __name__ == '__main__':
    args = parser.parse_args()

    class_reader = ClassificationReader(skip_start_end=True)
    class_vocab = Vocabulary.from_files(Path(args.classifier_path) / 'vocab')
    class_model_args = get_args_from_path(Path(args.classifier_path) / 'args.json')
    class_model = get_model_by_name(**class_model_args, vocab=class_vocab)
    load_weights(class_model, Path(args.classifier_path) / 'best.th')

    reader = CopyNetReader(masker=None)
    copynet_vocab = Vocabulary.from_files(Path(args.copynet_path) / 'vocab')
    copynet_model_args = get_args_from_path(Path(args.copynet_path) / 'args.json')
    copynet_model = get_model_by_name(
        **copynet_model_args,
        vocab=copynet_vocab,
        beam_size=args.beam_size
    )
    load_weights(copynet_model, Path(args.copynet_path) / 'best.th')

    class_model_copynet_args = get_args_from_path(Path(args.classifier_copynet_path) / 'args.json')
    class_model_copynet = get_model_by_name(
        **class_model_copynet_args,
        vocab=copynet_vocab,
        copynet=copynet_model
    )
    load_weights(class_model_copynet, Path(args.classifier_copynet_path) / 'best.th')

    deep_levenshtein_model_args = get_args_from_path(Path(args.levenshtein_path) / 'args.json')
    deep_levenshtein_model = get_model_by_name(
        **deep_levenshtein_model_args,
        vocab=copynet_vocab,
        copynet=copynet_model
    )
    load_weights(deep_levenshtein_model, Path(args.levenshtein_path) / 'best.th')

    attacker = Cascada(
        vocab=copynet_vocab,
        reader=reader,
        class_reader=class_reader,
        classification_model_basic=class_model,
        classification_model=class_model_copynet,
        masked_copynet=copynet_model, deep_levenshtein_model=deep_levenshtein_model,
        levenshtein_weight=args.levenshtein_weight,
        prob_diff_weight=args.prob_diff_weight,
        learning_rate=args.learning_rate,
        num_updates=args.num_updates,
        num_labels=class_model_args['num_classes'],
        device=args.cuda
    )

    data = pd.read_csv(args.csv_path)
    sequences = data['sequences'].tolist()[:args.sample]
    labels = data['labels'].tolist()[:args.sample]
    maskers = [args.maskers.split(',')] * len(sequences)

    results_path = Path(args.results_path) / datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path.mkdir(exist_ok=True, parents=True)
    path_to_results_file = results_path / 'results.csv'
    dump_metrics(results_path / 'args.json', args.__dict__)
    with open(path_to_results_file, 'w', newline='') as csv_write:
        fieldnames = list(AttackerOutput.__annotations__.keys())
        writer = csv.DictWriter(csv_write, fieldnames=fieldnames)
        writer.writeheader()
        for seq, lab, mask_tokens in tqdm(zip(sequences, labels, maskers)):

            attacker.set_label_to_attack(lab)
            attacker.set_input(sequence=seq, mask_tokens=mask_tokens)

            output = attacker.sample_until_label_is_changed(
                max_steps=args.max_steps,
                early_stopping=args.early_stopping
            ).__dict__

            attacker.empty_history()

            writer.writerow(output)
            csv_write.flush()
