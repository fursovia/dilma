import argparse
from pathlib import Path
from pprint import pprint

import torch
import torch.optim as optim
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer
from allennlp.data.iterators import BucketIterator
from allennlp.common.util import dump_metrics

from adat.models import get_basic_classification_model, \
    get_basic_seq2seq_model, get_mask_seq2seq_model, get_att_mask_seq2seq_model, get_basic_deep_levenshtein
from adat.dataset import CsvReader, OneLangSeq2SeqReader, Task, END_SYMBOL, START_SYMBOL, LevenshteinReader
from adat.masker import get_default_masker

LEARNING_RATE = 0.003
BEAM_SIZE = 1
MAX_DECODING_STEPS = 20

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


if __name__ == '__main__':
    args = parser.parse_args()

    if args.task == Task.CLASSIFICATION:
        reader = CsvReader(lazy=False)
        sorting_keys = [('tokens', 'num_tokens')]
    elif args.task == Task.SEQ2SEQ or args.task == Task.MASKEDSEQ2SEQ or args.task == Task.ATTMASKEDSEQ2SEQ:
        mask = get_default_masker() if args.use_mask else None
        reader = OneLangSeq2SeqReader(mask)
        sorting_keys = [('source_tokens', 'num_tokens')]
    elif args.task == Task.DEEPLEVENSHTEIN:
        reader = LevenshteinReader()
        sorting_keys = [('sequence_a', 'num_tokens'), ('sequence_b', 'num_tokens')]
    else:
        raise NotImplementedError(f'{args.task} -- no such task')

    data_path = Path(args.data_dir)
    train_dataset = reader.read(data_path / 'train.csv')
    test_dataset = reader.read(data_path / 'test.csv')

    vocab = Vocabulary.from_instances(
        train_dataset + test_dataset,
        tokens_to_add={'tokens': [START_SYMBOL, END_SYMBOL]}
    )

    iterator = BucketIterator(batch_size=args.batch_size, sorting_keys=sorting_keys)
    iterator.index_with(vocab)

    if args.task == Task.CLASSIFICATION:
        model = get_basic_classification_model(vocab, args.num_classes)
    elif args.task == Task.SEQ2SEQ:
        model = get_basic_seq2seq_model(vocab, max_decoding_steps=MAX_DECODING_STEPS, beam_size=BEAM_SIZE)
    elif args.task == Task.MASKEDSEQ2SEQ:
        model = get_mask_seq2seq_model(vocab, max_decoding_steps=MAX_DECODING_STEPS, beam_size=BEAM_SIZE)
    elif args.task == Task.ATTMASKEDSEQ2SEQ:
        model = get_att_mask_seq2seq_model(vocab, max_decoding_steps=MAX_DECODING_STEPS, beam_size=BEAM_SIZE)

    elif args.task == Task.DEEPLEVENSHTEIN:
        model = get_basic_deep_levenshtein(vocab)
    else:
        raise NotImplementedError(f'{args.task} -- no such task')

    if args.cuda >= 0 and torch.cuda.is_available():
        model.cuda(args.cuda)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(exist_ok=True)
    vocab.save_to_files(model_dir / "vocab")
    dump_metrics(model_dir / "args.json", args.__dict__)

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
