import argparse
from pathlib import Path
import csv
from tqdm import tqdm

import pandas as pd
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.util import dump_metrics

from adat.dataset import OneLangSeq2SeqReader, IDENTITY_TOKEN, Task
from adat.utils import load_weights
from adat.attackers import AttackerOutput, GradientAttacker
from adat.models import (
    get_deep_levenshtein,
    get_deep_levenshtein_att,
    get_classification_model,
    get_classification_model_seq2seq,
    get_deep_levenshtein_seq2seq,
    get_att_mask_seq2seq_model,
    get_seq2seq_model,
)

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=-1)
parser.add_argument('--csv_path', type=str, default='data/test.csv', required=True)
parser.add_argument('--results_path', type=str, default='results')
parser.add_argument('-sp', '--seq2seq_path', type=str, default='experiments/seq2seq')
parser.add_argument('-cp', '--classification_path', type=str, default='experiments/classification')
parser.add_argument('-lp', '--levenshtein_path', type=str, default='experiments/deep_levenshtein')
parser.add_argument('-lw', '--levenshtein_weight', type=float, default=0.1)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.5)
parser.add_argument('-nu', '--num_updates', type=int, default=15)
parser.add_argument('-bs', '--beam_size', type=int, default=1)
parser.add_argument('-m', '--maskers', type=str, default=IDENTITY_TOKEN)
parser.add_argument('-nc', '--num_classes', type=int, default=2)
parser.add_argument('--s2s_model', type=Task, default=Task.SEQ2SEQ, help='seq2seq model type')
parser.add_argument('--ed_model', type=Task, default=Task.DEEPLEVENSHTEIN, help='edit distance model type')
parser.add_argument('--clf_model', type=Task, default=Task.CLASSIFICATION, help='classification model type')

if __name__ == '__main__':
    args = parser.parse_args()

    reader = OneLangSeq2SeqReader()
    vocab_path = Path(args.seq2seq_path) / 'vocab'
    vocab = Vocabulary.from_files(vocab_path)

    if args.s2s_model == Task.ATTMASKEDSEQ2SEQ:
        seq2seq_model = get_att_mask_seq2seq_model(vocab, beam_size=args.beam_size)
    elif args.s2s_model == Task.SEQ2SEQ:
        seq2seq_model = get_seq2seq_model(vocab, beam_size=args.beam_size)
        
    #classification_model = get_classification_model_seq2seq(seq2seq_model, args.num_classes)
    if args.clf_model == Task.CLASSIFICATION:
        classification_model = get_classification_model(vocab, args.num_classes)
    elif args.clf_model == Task.CLASSIFICATIONSEQ2SEQ:
        classification_model = get_classification_model_seq2seq(seq2seq_model, args.num_classes)
    
    load_weights(seq2seq_model, Path(args.seq2seq_path) / 'best.th')
    load_weights(classification_model, Path(args.classification_path) / 'best.th')
    
    #levenshtein_model = get_deep_levenshtein_seq2seq(seq2seq_model)
    if args.ed_model == Task.DEEPLEVENSHTEIN:
        levenshtein_model = get_deep_levenshtein(vocab)
    elif args.ed_model == Task.DEEPLEVENSHTEINSEQ2SEQ:
        levenshtein_model = get_deep_levenshtein_seq2seq(seq2seq_model)
    elif args.ed_model == Task.DEEPLEVENSHTEINATT:
        levenshtein_model = get_deep_levenshtein_att(vocab)

    load_weights(levenshtein_model, Path(args.levenshtein_path) / 'best.th')

    attacker = GradientAttacker(
        vocab=vocab,
        reader=reader,
        classification_model=classification_model,
        seq2seq_model=seq2seq_model,
        deep_levenshtein_model=levenshtein_model,
        levenshtein_weight=args.levenshtein_weight,
        device=args.cuda,
        num_labels=args.num_classes
    )

    data = pd.read_csv(args.csv_path)
    sequences = data['sequences'].tolist()
    labels = data['labels'].tolist()
    maskers = [args.maskers.split()] * len(sequences)

    results_path = Path(args.results_path)
    results_path.mkdir(exist_ok=True)
    path_to_results_file = results_path / 'results.csv'
    dump_metrics(results_path / 'args.json', args.__dict__)
    #assert not path_to_results_file.exists(), \
    #    f'You already have `{path_to_results_file}` file. Delete it or change --results_path.'
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
