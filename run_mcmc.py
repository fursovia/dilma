import argparse
from pathlib import Path
import csv
from tqdm import tqdm
from datetime import datetime

import pandas as pd
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.util import dump_metrics

from adat.dataset import ClassificationReader, CopyNetReader, IDENTITY_TOKEN
from adat.attackers import MCMCSampler, RandomSampler, NormalProposal, AttackerOutput
from adat.models import get_model_by_name
from adat.utils import load_weights, get_args_from_path

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=-1)
parser.add_argument('--csv_path', type=str, default='data/test.csv')
parser.add_argument('--results_path', type=str, default='results')

parser.add_argument('--copynet_path', type=str, default='experiments/copynet')
parser.add_argument('--classifier_path', type=str, default='experiments/classification')
parser.add_argument('--random', action='store_true', help='Whether to use RandomSampler instead of MCMC')

parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--beam_size', type=int, default=1)
parser.add_argument('--std', type=float, default=0.01)
parser.add_argument('--sigma_class', type=float, default=1.0)
parser.add_argument('--sigma_wer', type=float, default=0.5)
parser.add_argument('--maskers', type=str, default=IDENTITY_TOKEN, help='string with comma-separated values')
parser.add_argument('--space', type=str, default='decoder_hidden')
parser.add_argument('--early_stopping', action='store_true')
parser.add_argument('--sample', type=int, default=None)


if __name__ == '__main__':
    args = parser.parse_args()

    class_reader = ClassificationReader(skip_start_end=True)
    class_vocab = Vocabulary.from_files(Path(args.classifier_path) / 'vocab')
    class_model_args = get_args_from_path(Path(args.classifier_path) / 'args.json')
    class_model = get_model_by_name(**class_model_args, vocab=class_vocab)
    load_weights(class_model, Path(args.classifier_path) / 'best.th')

    copynet_reader = CopyNetReader(masker=None)
    copynet_vocab = Vocabulary.from_files(Path(args.copynet_path) / 'vocab')
    copynet_model_args = get_args_from_path(Path(args.copynet_path) / 'args.json')
    copynet_model = get_model_by_name(**copynet_model_args, vocab=copynet_vocab, beam_size=args.beam_size)
    load_weights(copynet_model, Path(args.copynet_path) / 'best.th')

    if args.random:
        sampler = RandomSampler(
            proposal_distribution=NormalProposal(scale=args.std),
            classification_model=class_model,
            classification_reader=class_reader,
            generation_model=copynet_model,
            generation_reader=copynet_reader,
            space=args.space,
            device=args.cuda
        )
    else:
        sampler = MCMCSampler(
            proposal_distribution=NormalProposal(scale=args.std),
            classification_model=class_model,
            classification_reader=class_reader,
            generation_model=copynet_model,
            generation_reader=copynet_reader,
            sigma_class=args.sigma_class,
            sigma_wer=args.sigma_wer,
            space=args.space,
            device=args.cuda
        )

    data = pd.read_csv(args.csv_path)
    sequences = data['sequences'].tolist()[:args.sample]
    labels = data['labels'].tolist()[:args.sample]
    maskers = [args.maskers.split(',')] * len(sequences)  # can be unique for each sequence

    results_path = Path(args.results_path) / datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path.mkdir(exist_ok=True, parents=True)
    path_to_results_file = results_path / 'results.csv'
    dump_metrics(results_path / 'args.json', args.__dict__)
    with open(path_to_results_file, 'w', newline='') as csv_write:
        fieldnames = list(AttackerOutput.__annotations__.keys())
        writer = csv.DictWriter(csv_write, fieldnames=fieldnames)
        writer.writeheader()
        for seq, lab, mask in tqdm(zip(sequences, labels, maskers)):
            sampler.set_label_to_attack(lab)
            sampler.set_input(seq, mask_tokens=mask)
            output = sampler.sample_until_label_is_changed(
                max_steps=args.num_steps,
                early_stopping=args.early_stopping
            ).__dict__
            sampler.empty_history()

            writer.writerow(output)
            csv_write.flush()
