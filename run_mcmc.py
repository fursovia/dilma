import argparse
from pathlib import Path
import csv
from tqdm import tqdm

import numpy as np
from allennlp.data.vocabulary import Vocabulary
from adat.dataset import CsvReader, OneLangSeq2SeqReader

from adat.mcmc import MCMCSampler, NormalProposal
from adat.models import get_basic_classification_model, get_basic_seq2seq_model
from adat.utils import load_weights
from train import MAX_DECODING_STEPS

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=-1)
parser.add_argument('--csv_path', type=str, default='data/test.csv')
parser.add_argument('--results_path', type=str, default='results')
parser.add_argument('-cd', '--class_dir', type=str, default='experiments/classification')
parser.add_argument('-ssd', '--seq2seq_dir', type=str, default='experiments/seq2seq')
parser.add_argument('-ns', '--num_steps', type=int, default=200)
parser.add_argument('-bs', '--beam_size', type=int, default=1)
parser.add_argument('--var', type=float, default=0.01)
parser.add_argument('--sigma_prob', type=float, default=1.0)
parser.add_argument('--sigma_bleu', type=float, default=1.0)
parser.add_argument('--verbose', type=int, default=100)


if __name__ == '__main__':
    args = parser.parse_args()

    class_reader = CsvReader()
    class_vocab = Vocabulary.from_files(Path(args.class_dir) / 'vocab')
    class_model = get_basic_classification_model(class_vocab)
    load_weights(class_model, Path(args.class_dir) / 'best.th')

    seq2seq_reader = OneLangSeq2SeqReader(masker=None)
    seq2seq_vocab = Vocabulary.from_files(Path(args.seq2seq_dir) / 'vocab')
    seq2seq_model = get_basic_seq2seq_model(seq2seq_vocab, max_decoding_steps=MAX_DECODING_STEPS,
                                            beam_size=args.beam_size)
    load_weights(seq2seq_model, Path(args.seq2seq_dir) / 'best.th')

    sampler = MCMCSampler(
        proposal_distribution=NormalProposal(args.var),
        classification_model=class_model,
        classification_reader=class_reader,
        generation_model=seq2seq_model,
        generation_reader=seq2seq_reader,
        sigma_prob=args.sigma_prob,
        sigma_bleu=args.sigma_bleu,
        device=args.cuda
    )

    results_path = Path(args.results_path)
    results_path.mkdir(exist_ok=True)
    with open(args.csv_path, "r") as csv_read, open(results_path / 'results.csv', 'w', newline='') as csv_write:
        fieldnames = ['generated_sequence', 'prob', 'bleu', 'prob_diff', 'prob_drop', 'bleu_diff',
                      'bleu_drop', 'acceptance_probability', 'seq_len', 'original']
        writer = csv.DictWriter(csv_write, fieldnames=fieldnames)
        writer.writeheader()

        tsv_in = csv.reader(csv_read, delimiter=',')
        next(tsv_in, None)
        next(tsv_in, None)
        for row in tqdm(tsv_in):
            example = row[0].strip()
            curr_label = int(row[1].strip())

            sampler.set_label_to_drop(curr_label)
            sampler.set_input(example)
            generated_seq = sampler.sample_until_satisfied(max_steps=args.num_steps)
            if generated_seq is not None:
                ex = sampler.history[-1]
            else:
                ex = {
                    'generated_sequence': 'None',
                    'prob': 1.0,
                    'bleu': 0.0,
                    'prob_diff': 0.0,
                    'prob_drop': 0.0,
                    'bleu_diff': -1.0,
                    'bleu_drop': np.inf,
                    'acceptance_probability': 0.0
                }

            ex['seq_len'] = len(example.split())
            ex['original'] = example
            sampler.empty_history()
            writer.writerow(ex)
