
import torch
import torch.optim as optim
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers import SimpleLanguageModelingDatasetReader
from allennlp.training.trainer import Trainer
from allennlp.data.iterators import BasicIterator

from adat.lm import get_basic_lm


TRAIN_PATH = '/Users/fursovia/Documents/texar/examples/text_style_transfer/data/insurance_cropped/insurance.train.text'
TEST_PATH = '/Users/fursovia/Documents/texar/examples/text_style_transfer/data/insurance_cropped/insurance.test.text'


reader = SimpleLanguageModelingDatasetReader()
train_dataset = reader.read(TRAIN_PATH)
test_dataset = reader.read(TEST_PATH)

vocab = Vocabulary.from_instances(train_dataset)

iterator = BasicIterator(batch_size=64)
iterator.index_with(vocab)

model = get_basic_lm(vocab)
optimizer = optim.Adam(model.parameters(), lr=0.001)


trainer = Trainer(
    model=model,
    optimizer=optimizer,
    iterator=iterator,
    train_dataset=train_dataset,
    validation_dataset=test_dataset,
    patience=3,
    num_epochs=10,
)

results = trainer.train()
print('Results')
print(results)

with open("model.th", 'wb') as f:
    torch.save(model.state_dict(), f)

vocab.save_to_files("vocab")