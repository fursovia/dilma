import argparse
from pathlib import Path
import csv
from tqdm import tqdm

import pandas as pd
import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.util import dump_metrics

from adat.dataset import CopyNetReader, IDENTITY_TOKEN
from adat.utils import load_weights, get_args_from_path
from adat.attackers import AttackerOutput
from adat.attackers.cascada import Cascada
from adat.models import get_model_by_name


parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=-1)
parser.add_argument('--csv_path', type=str, default='data/test.csv', required=True)
parser.add_argument('--results_path', type=str, default='results')

parser.add_argument('--copynet_path', type=str, default='experiments/copynet')
parser.add_argument('--classifier_path', type=str, default='experiments/classification')
parser.add_argument('--levenshtein_path', type=str, default='experiments/deep_levenshtein')

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

    reader = CopyNetReader(masker=None)

    copynet_vocab = Vocabulary.from_files(Path(args.copynet_path) / 'vocab')
    copynet_model_args = get_args_from_path(Path(args.copynet_path) / 'args.json')
    copynet_model = get_model_by_name(
        **copynet_model_args,
        vocab=copynet_vocab,
        beam_size=args.beam_size
    )
    load_weights(copynet_model, Path(args.copynet_path) / 'best.th')

    class_model_args = get_args_from_path(Path(args.classifier_path) / 'args.json')
    class_model = get_model_by_name(
        **class_model_args,
        vocab=copynet_vocab,
        copynet=copynet_model
    )
    load_weights(class_model, Path(args.classifier_path) / 'best.th')

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
        classification_model=class_model,
        masked_copynet=copynet_model, deep_levenshtein_model=deep_levenshtein_model,
        levenshtein_weight=args.levenshtein_weight,
        learning_rate=args.learning_rate,
        num_updates=args.num_updates,
        num_labels=class_model_args['num_classes'],
        device=args.cuda
    )

    data = pd.read_csv(args.csv_path)
    sequences = data['sequences'].tolist()[:args.sample]
    labels = data['labels'].tolist()[:args.sample]
    maskers = [args.maskers.split(',')] * len(sequences)

    results_path = Path(args.results_path)
    results_path.mkdir(exist_ok=True, parents=True)
    path_to_results_file = results_path / 'results.csv'
    dump_metrics(results_path / 'args.json', args.__dict__)
    # assert not path_to_results_file.exists(), \
    #     f'You already have `{path_to_results_file}` file. Delete it or change --results_path.'
    with open(path_to_results_file, 'w', newline='') as csv_write:
        fieldnames = list(AttackerOutput.__annotations__.keys())
        writer = csv.DictWriter(csv_write, fieldnames=fieldnames)
        writer.writeheader()
        for seq, lab, mask_tokens in tqdm(zip(sequences, labels, maskers)):

            # it's important to do it under `no_grad()`
            with torch.no_grad():
                attacker.set_input(sequence=seq, mask_tokens=mask_tokens)
                attacker.set_label_to_attack(lab)

            output = attacker.sample_until_label_is_changed(
                max_steps=args.max_steps,
                early_stopping=args.early_stopping
            ).__dict__

            attacker.empty_history()

            writer.writerow(output)
            csv_write.flush()
