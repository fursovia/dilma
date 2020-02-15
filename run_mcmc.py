import argparse
from pathlib import Path
import csv
import json
from tqdm import tqdm

import pandas as pd
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.util import dump_metrics

from adat.dataset import ClassificationReader, CopyNetReader
from adat.attackers.mcmc import MCMCSampler, RandomSampler, NormalProposal, SamplerOutput
from adat.models import Task, get_model_by_name
from adat.utils import load_weights

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=-1)
parser.add_argument('--csv_path', type=str, default='data/test.csv')
parser.add_argument('--results_path', type=str, default='results')
parser.add_argument('--classification_path', type=str, default='experiments/classification')
parser.add_argument('--seq2seq_path', type=str, default='experiments/seq2seq')
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--beam_size', type=int, default=1)
parser.add_argument('--std', type=float, default=0.01)
parser.add_argument('--sigma_class', type=float, default=1.0)
parser.add_argument('--sigma_wer', type=float, default=0.5)
parser.add_argument('--maximum_wer', type=float, default=0.2)
parser.add_argument('--minimum_prob_drop', type=float, default=2.0)
parser.add_argument('--random', action='store_true', help='Whether to use RandomSampler instead of MCMC')
parser.add_argument('--sample', type=int, default=None)


def _get_classifier_from_args(vocab: Vocabulary, path: str):
    with open(path) as file:
        args = json.load(file)
    num_classes = args['num_classes']
    return get_classification_model(vocab, int(num_classes))


def _get_seq2seq_from_args(vocab: Vocabulary, path: str, beam_size: int):
    with open(path) as file:
        args = json.load(file)
    task = args['task']
    use_attention = args['no_attention']
    if task == Task.SEQ2SEQ:
        return get_seq2seq_model(vocab, beam_size=beam_size, use_attention=use_attention)
    elif task == Task.ATTMASKEDSEQ2SEQ:
        # TODO: unstable
        return get_att_mask_seq2seq_model(vocab, beam_size=beam_size, use_attention=use_attention)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    args = parser.parse_args()

    class_reader = ClassificationReader(skip_start_end=True)
    class_vocab = Vocabulary.from_files(Path(args.classification_path) / 'vocab')
    class_model = _get_classifier_from_args(class_vocab, Path(args.classification_path) / 'args.json')
    load_weights(class_model, Path(args.classification_path) / 'best.th')

    seq2seq_reader = CopyNetReader(masker=None)
    seq2seq_vocab = Vocabulary.from_files(Path(args.seq2seq_path) / 'vocab')
    seq2seq_model = _get_seq2seq_from_args(
        seq2seq_vocab,
        Path(args.seq2seq_path) / 'args.json',
        beam_size=args.beam_size
    )
    load_weights(seq2seq_model, Path(args.seq2seq_path) / 'best.th')

    if args.random:
        sampler = RandomSampler(
            proposal_distribution=NormalProposal(args.std),
            classification_model=class_model,
            classification_reader=class_reader,
            generation_model=seq2seq_model,
            generation_reader=seq2seq_reader,
            device=args.cuda
        )
    else:
        sampler = MCMCSampler(
            proposal_distribution=NormalProposal(args.std),
            classification_model=class_model,
            classification_reader=class_reader,
            generation_model=seq2seq_model,
            generation_reader=seq2seq_reader,
            sigma_class=args.sigma_class,
            sigma_wer=args.sigma_wer,
            device=args.cuda
        )

    data = pd.read_csv(args.csv_path)
    sequences = data['sequences'].tolist()[:args.sample]
    labels = data['labels'].tolist()[:args.sample]

    results_path = Path(args.results_path)
    results_path.mkdir(exist_ok=True)
    path_to_results_file = results_path / 'results.csv'
    assert not path_to_results_file.exists(), \
        f'You already have `{path_to_results_file}` file. Delete it or change --results_path.'
    dump_metrics(results_path / 'args.json', args.__dict__)
    with open(path_to_results_file, 'w', newline='') as csv_write:
        fieldnames = list(SamplerOutput.__annotations__.keys())
        writer = csv.DictWriter(csv_write, fieldnames=fieldnames)
        writer.writeheader()
        for seq, lab in tqdm(zip(sequences, labels)):
            sampler.set_label_to_attack(lab)
            sampler.set_input(seq)
            ex = sampler.sample_until_satisfied(
                max_steps=args.num_steps,
                wer=args.maximum_wer,
                prob_drop=args.minimum_prob_drop
            ).__dict__
            sampler.empty_history()

            writer.writerow(ex)
            csv_write.flush()
