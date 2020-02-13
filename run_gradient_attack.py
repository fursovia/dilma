import argparse
from pathlib import Path
import csv
import json
from typing import List
from tqdm import tqdm

import pandas as pd
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.util import dump_metrics

from adat.dataset import OneLangSeq2SeqReader, IDENTITY_TOKEN, Task
from adat.utils import load_weights
from adat.attackers import AttackerOutput, GradientAttacker
from adat.models import (
    get_seq2seq_model,
    get_classification_model_seq2seq,
    get_deep_levenshtein_seq2seq,
    get_att_mask_seq2seq_model
)

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=-1)
parser.add_argument('--csv_path', type=str, default='data/test.csv', required=True)
parser.add_argument('--results_path', type=str, default='results')
parser.add_argument('-sp', '--seq2seq_path', type=str, default='experiments/seq2seq')
parser.add_argument('-cp', '--classification_path', type=str, default='experiments/classification')
parser.add_argument('-lp', '--levenshtein_path', type=str, default='experiments/deep_levenshtein')
parser.add_argument('-lw', '--levenshtein_weight', type=float, default=0.1)
parser.add_argument('-lr', '--learning_rate', type=float, default=2.0)
parser.add_argument('-nu', '--num_updates', type=int, default=50)
parser.add_argument('-bs', '--beam_size', type=int, default=1)
parser.add_argument('-m', '--maskers', type=str, default=IDENTITY_TOKEN, help='string with comma-separated values')
parser.add_argument('--sample', type=int, default=None)


def _get_classifier_from_args(seq2seq, path: str):
    with open(path) as file:
        args = json.load(file)
    num_classes = args['num_classes']
    return get_classification_model_seq2seq(seq2seq, int(num_classes)), int(num_classes)


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


def get_heuristics_maskers(seq: str) -> List[str]:
    length = len(seq.split())
    if length <= 2:
        maskers = ['RemoveMasker']
    elif length >= 3 and length <= 5:
        maskers = ['RemoveMasker', 'ReplaceMasker']
    elif length >= 6 and length <= 8:
        maskers = ['ReplaceMasker', 'AddMaskerrandom']
    else:
        maskers = ['ReplaceMasker', 'ReplaceMasker', 'AddMaskerrandom', 'AddMaskerrandom']

    return maskers


if __name__ == '__main__':
    args = parser.parse_args()

    reader = OneLangSeq2SeqReader()
    vocab_path = Path(args.seq2seq_path) / 'vocab'
    vocab = Vocabulary.from_files(vocab_path)

    seq2seq_model = _get_seq2seq_from_args(vocab, Path(args.seq2seq_path) / 'args.json', args.beam_size)
    classification_model, num_classes = _get_classifier_from_args(
        seq2seq_model,
        Path(args.classification_path) / 'args.json'
    )
    levenshtein_model = get_deep_levenshtein_seq2seq(seq2seq_model)

    load_weights(seq2seq_model, Path(args.seq2seq_path) / 'best.th')
    load_weights(classification_model, Path(args.classification_path) / 'best.th')
    load_weights(levenshtein_model, Path(args.levenshtein_path) / 'best.th')

    attacker = GradientAttacker(
        vocab=vocab,
        reader=reader,
        classification_model=classification_model,
        seq2seq_model=seq2seq_model,
        deep_levenshtein_model=levenshtein_model,
        levenshtein_weight=args.levenshtein_weight,
        device=args.cuda,
        num_labels=num_classes
    )

    data = pd.read_csv(args.csv_path)
    sequences = data['sequences'].tolist()[:args.sample]
    labels = data['labels'].tolist()[:args.sample]
    if args.maskers != 'auto':
        maskers_to_use = args.maskers.split(',')
        print(f'MASKERS = {maskers_to_use}')
        maskers = [maskers_to_use] * len(sequences)
    else:
        maskers = [get_heuristics_maskers(seq) for seq in sequences]

    results_path = Path(args.results_path)
    results_path.mkdir(exist_ok=True)
    path_to_results_file = results_path / 'results.csv'
    dump_metrics(results_path / 'args.json', args.__dict__)
    assert not path_to_results_file.exists(), \
        f'You already have `{path_to_results_file}` file. Delete it or change --results_path.'
    with open(path_to_results_file, 'w', newline='') as csv_write:
        fieldnames = list(AttackerOutput.__annotations__.keys())
        writer = csv.DictWriter(csv_write, fieldnames=fieldnames)
        writer.writeheader()
        for seq, lab, masker in tqdm(zip(sequences, labels, maskers)):
            output = attacker.attack(
                sequences=[seq],
                labels=[lab],
                maskers=[masker],
                learning_rate=args.learning_rate,
                num_updates=args.num_updates
            )[0]
            writer.writerow(output.__dict__)
            csv_write.flush()
