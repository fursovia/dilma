import argparse
import json
from pathlib import Path
from pprint import pprint

import torch
import torch.optim as optim
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer
from allennlp.data.iterators import BucketIterator
from allennlp.common.util import dump_metrics

from adat.models import (
    OneLanguageSeq2SeqModel,
    get_classification_model,
    get_classification_model_seq2seq,
    get_seq2seq_model,
    get_att_mask_seq2seq_model,
    get_deep_levenshtein,
    get_deep_levenshtein_seq2seq,
    get_deep_levenshtein_att
)
from adat.dataset import CsvReader, OneLangSeq2SeqReader, Task, END_SYMBOL, START_SYMBOL, LevenshteinReader
from adat.masker import get_default_masker
from adat.utils import load_weights

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=-1, help='cuda device number')
parser.add_argument('--task', type=Task, default=Task.CLASSIFICATION, help='task name')
parser.add_argument('-md', '--model_dir', type=str, default='experiments', help='where to save checkpoints')
parser.add_argument('-dd', '--data_dir', type=str, default='data', help='where train.csv and test.csv are')
parser.add_argument('-ne', '--num_epochs', type=int, default=10)
parser.add_argument('-bs', '--batch_size', type=int, default=1024)
parser.add_argument('-p', '--patience', type=int, default=2,
                    help='Number of epochs to be patient before early stopping')
parser.add_argument('-nc', '--num_classes', type=int, default=2)
parser.add_argument('-um', '--use_mask', action='store_true', help='Whether to apply masking to the input')
parser.add_argument('-s2smd', '--seq2seq_model_dir', type=str, default=None)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--no_attention', action='store_false')


LEARNING_RATE = 0.003
MAX_DECODING_STEPS = 20
BEAM_SIZE = 1


def get_specific_seq2seq(task: Task, vocab: Vocabulary,
                         max_decoding_steps: int = MAX_DECODING_STEPS,
                         beam_size: int = BEAM_SIZE, use_attention: bool = True) -> OneLanguageSeq2SeqModel:
    if task == Task.SEQ2SEQ:
        return get_seq2seq_model(vocab, max_decoding_steps, beam_size, use_attention)
    elif task == Task.ATTMASKEDSEQ2SEQ:
        return get_att_mask_seq2seq_model(vocab, max_decoding_steps, beam_size, use_attention)
    else:
        raise NotImplementedError


def get_seq2seq_task_name(path: str) -> str:
    with open(Path(path) / 'args.json') as file:
        return json.load(file)['task']


if __name__ == '__main__':
    args = parser.parse_args()

    # DATASETS
    if args.task in [Task.CLASSIFICATION, Task.CLASSIFICATIONSEQ2SEQ]:
        reader = CsvReader(lazy=False)
        test_reader = reader
        sorting_keys = [('tokens', 'num_tokens')]
    elif args.task in [Task.SEQ2SEQ, Task.ATTMASKEDSEQ2SEQ]:
        mask = get_default_masker() if args.use_mask else None
        reader = OneLangSeq2SeqReader(mask)
        test_reader = OneLangSeq2SeqReader(masker=None)
        sorting_keys = [('source_tokens', 'num_tokens')]
    elif args.task in [Task.DEEPLEVENSHTEIN, Task.DEEPLEVENSHTEINSEQ2SEQ, Task.DEEPLEVENSHTEINATT]:
        reader = LevenshteinReader()
        test_reader = reader
        sorting_keys = [('sequence_a', 'num_tokens'), ('sequence_b', 'num_tokens')]
    else:
        raise NotImplementedError(f'{args.task} -- no such task')

    data_path = Path(args.data_dir)
    train_dataset = reader.read(data_path / 'train.csv')
    test_dataset = test_reader.read(data_path / 'test.csv')

    model_dir = Path(args.model_dir)
    model_dir.mkdir(exist_ok=True)
    dump_metrics(model_dir / "args.json", args.__dict__)

    # VOCABULARY, ITERATOR
    if args.resume:
        vocab = Vocabulary.from_files(model_dir / "vocab")
    elif args.seq2seq_model_dir is not None:
        vocab = Vocabulary.from_files(Path(args.seq2seq_model_dir) / "vocab")
    else:
        vocab = Vocabulary.from_instances(
            train_dataset + test_dataset,
            tokens_to_add={'tokens': [START_SYMBOL, END_SYMBOL]}
        )
        vocab.save_to_files(model_dir / "vocab")

    iterator = BucketIterator(batch_size=args.batch_size, sorting_keys=sorting_keys)
    iterator.index_with(vocab)

    # MODELS
    if args.task == Task.CLASSIFICATION:
        model = get_classification_model(vocab, args.num_classes)
    elif args.task == Task.CLASSIFICATIONSEQ2SEQ:
        seq2seq_model = get_specific_seq2seq(get_seq2seq_task_name(args.seq2seq_model_dir), vocab,
                                             use_attention=args.no_attention)
        load_weights(seq2seq_model, Path(args.seq2seq_model_dir) / 'best.th')
        model = get_classification_model_seq2seq(seq2seq_model, args.num_classes)
    elif args.task in [Task.SEQ2SEQ, Task.ATTMASKEDSEQ2SEQ]:
        model = get_specific_seq2seq(args.task, vocab,
                                     use_attention=args.no_attention)
    elif args.task == Task.DEEPLEVENSHTEIN:
        model = get_deep_levenshtein(vocab)
    elif args.task == Task.DEEPLEVENSHTEINSEQ2SEQ:
        seq2seq_model = get_specific_seq2seq(get_seq2seq_task_name(args.seq2seq_model_dir), vocab,
                                             use_attention=args.no_attention)
        load_weights(seq2seq_model, Path(args.seq2seq_model_dir) / 'best.th')
        model = get_deep_levenshtein_seq2seq(seq2seq_model)
    elif args.task == Task.DEEPLEVENSHTEINATT:
        model = get_deep_levenshtein_att(vocab)
    else:
        raise NotImplementedError(f'{args.task} -- no such task')

    if args.resume:
        load_weights(model, model_dir / 'best.th')

    if args.cuda >= 0 and torch.cuda.is_available():
        model.cuda(args.cuda)

    # TRAINING
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=test_dataset,
        serialization_dir=model_dir,
        patience=args.patience,
        num_epochs=args.num_epochs,
        cuda_device=args.cuda
    )

    results = trainer.train()
    pprint(results)
