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


TRAIN_PATH = '/Users/fursovia/Documents/texar/examples/text_style_transfer/data/insurance_cropped/insurance.train.text'
TEST_PATH = '/Users/fursovia/Documents/texar/examples/text_style_transfer/data/insurance_cropped/insurance.test.text'
BATCH_SIZE = 256
NUM_EPOCHS = 20
PATIENCE = 3


parser = argparse.ArgumentParser()
parser.add_argument('-md', '--model_dir', type=str, default='experiments')
parser.add_argument('-dd', '--data_dir', type=str, default='data')


if __name__ == '__main__':
    args = parser.parse_args()

    reader = SimpleLanguageModelingDatasetReader(
        tokenizer=WhitespaceTokenizer(),
        max_sequence_length=None
    )
    train_dataset = reader.read(TRAIN_PATH)
    test_dataset = reader.read(TEST_PATH)

    vocab = Vocabulary.from_instances(train_dataset)

    iterator = BasicIterator(batch_size=BATCH_SIZE)
    iterator.index_with(vocab)

    model = get_basic_lm(vocab)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=test_dataset,
        patience=PATIENCE,
        num_epochs=NUM_EPOCHS,
    )

    results = trainer.train()
    print(results)

    model_path = Path(args.model_dir)
    with open(model_path / "model.th", 'wb') as f:
        torch.save(model.state_dict(), f)

    vocab.save_to_files(model_path / "vocab")
