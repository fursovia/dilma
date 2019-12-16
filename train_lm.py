import argparse
from pathlib import Path

import torch
import torch.optim as optim
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers import SimpleLanguageModelingDatasetReader
from allennlp.training.trainer import Trainer
from allennlp.data.iterators import BasicIterator

from adat.lm import get_basic_lm
from adat.dataset import WhitespaceTokenizer


BATCH_SIZE = 512
NUM_EPOCHS = 10
PATIENCE = 2


parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=-1)
parser.add_argument('-md', '--model_dir', type=str, default='experiments')
parser.add_argument('-dd', '--data_dir', type=str, default='data')


if __name__ == '__main__':
    args = parser.parse_args()

    reader = SimpleLanguageModelingDatasetReader(
        tokenizer=WhitespaceTokenizer(),
        max_sequence_length=None
    )
    data_path = Path(args.data_dir)
    train_dataset = reader.read(data_path / 'train.txt')
    test_dataset = reader.read(data_path / 'test.txt')

    vocab = Vocabulary.from_instances(train_dataset)

    iterator = BasicIterator(batch_size=BATCH_SIZE)
    iterator.index_with(vocab)

    model = get_basic_lm(vocab)
    if args.cuda >= 0 and torch.cuda.is_available():
        model.cuda(args.cuda)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=test_dataset,
        patience=PATIENCE,
        num_epochs=NUM_EPOCHS,
        cuda_device=args.cuda
    )

    results = trainer.train()
    print(results)

    model_path = Path(args.model_dir)
    if not model_path.exists():
        model_path.mkdir()

    with open(model_path / "model.th", 'wb') as f:
        torch.save(model.state_dict(), f)

    vocab.save_to_files(model_path / "vocab")
