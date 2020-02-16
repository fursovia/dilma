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

from adat.models import Task, get_model_by_name
from adat.dataset import (
    ClassificationReader,
    CopyNetReader,
    LevenshteinReader,
    END_SYMBOL,
    START_SYMBOL
)
from adat.masker import get_default_masker
from adat.utils import load_weights

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=-1, help='cuda device number')
parser.add_argument('--task', type=Task, default=Task.CLASSIFICATION, help='task name')
parser.add_argument('-md', '--model_dir', type=str, default='experiments', help='where to save checkpoints')
parser.add_argument('-dd', '--data_dir', type=str, default='data', help='where train.csv and test.csv are')
parser.add_argument('--num_epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--beam_size', type=int, default=1)
parser.add_argument('--max_decoding_steps', type=int, default=20)
parser.add_argument('--patience', type=int, default=2,
                    help='Number of epochs to be patient before early stopping')
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--use_mask', action='store_true', help='Whether to apply masking to the input')
parser.add_argument('--copynet_dir', type=str, default=None)
parser.add_argument('--resume', action='store_true')


def _get_copynet_task_name(path: str) -> str:
    with open(Path(path) / 'json') as file:
        return json.load(file)['task']


if __name__ == '__main__':
    args = parser.parse_args()

    # DATASETS
    if args.task in [Task.CLASSIFICATION, Task.CLASSIFICATION_COPYNET]:
        skip_start_end = args.task == Task.CLASSIFICATION
        reader = ClassificationReader(lazy=False, skip_start_end=skip_start_end)
        sorting_keys = [('tokens', 'num_tokens')]
    elif args.task in [Task.NONMASKED_COPYNET, Task.NONMASKED_COPYNET_WITH_ATTENTION,
                       Task.MASKED_COPYNET_WITH_ATTNETION]:
        mask = get_default_masker() if args.use_mask else None
        reader = CopyNetReader(mask)
        sorting_keys = [('source_tokens', 'num_tokens')]
    elif args.task in [Task.DEEP_LEVENSHTEIN, Task.DEEP_LEVENSHTEIN_COPYNET, Task.DEEP_LEVENSHTEIN_WITH_ATTENTION]:
        reader = LevenshteinReader()
        sorting_keys = [('sequence_a', 'num_tokens'), ('sequence_b', 'num_tokens')]
    else:
        raise NotImplementedError(f'{args.task} -- no such task')

    data_path = Path(args.data_dir)
    train_dataset = reader.read(data_path / 'train.csv')
    test_dataset = reader.read(data_path / 'test.csv')

    model_dir = Path(args.model_dir)
    model_dir.mkdir(exist_ok=True, parents=True)
    dump_metrics(model_dir / "args.json", args.__dict__)

    # VOCABULARY, ITERATOR
    if args.resume:
        vocab = Vocabulary.from_files(model_dir / "vocab")
    elif args.copynet_dir is not None:
        vocab = Vocabulary.from_files(Path(args.copynet_dir) / "vocab")
    else:
        vocab = Vocabulary.from_instances(
            train_dataset + test_dataset,
            tokens_to_add={'tokens': [START_SYMBOL, END_SYMBOL]}
        )
        vocab.save_to_files(model_dir / "vocab")

    iterator = BucketIterator(batch_size=args.batch_size, sorting_keys=sorting_keys)
    iterator.index_with(vocab)

    if args.copynet_dir is not None:
        copynet = get_model_by_name(_get_copynet_task_name(args.copynet_dir), vocab)
        load_weights(copynet, Path(args.copynet_dir) / 'best.th')
    else:
        copynet = None

    model = get_model_by_name(
        args.task,
        vocab=vocab,
        num_classes=args.num_classes,
        beam_size=args.beam_size,
        max_decoding_steps=args.max_decoding_steps,
        copynet=copynet
    )

    if args.resume:
        load_weights(model, model_dir / 'best.th')

    if args.cuda >= 0 and torch.cuda.is_available():
        model.cuda(args.cuda)

    # TRAINING
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
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
