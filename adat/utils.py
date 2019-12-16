from typing import List

import torch
from allennlp.models import Model
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.training.metrics import Perplexity


def load_weights(model: Model, path: str, location: str = 'cpu') -> None:
    with open(path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=location))


def calculate_perplexity(texts: List[str], model: Model, reader: DatasetReader, vocab: Vocabulary) -> float:
    iterator = BasicIterator(batch_size=128)
    iterator.index_with(vocab)

    text_instances = [reader.text_to_instance(t) for t in texts]

    perplexity = Perplexity()

    for i, x in enumerate(iterator(text_instances, num_epochs=1)):
        with torch.no_grad():
            average_loss = model(**x)['loss']
            perplexity(average_loss)

    return perplexity.get_metric()
