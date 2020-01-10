import ast
from typing import List, Dict, Any

import torch
from allennlp.models import Model
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.training.metrics import Perplexity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import Levenshtein as lvs


def load_weights(model: Model, path: str, location: str = 'cpu') -> None:
    with open(path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=location))


def read_logs(results_path: str) -> List[Dict[str, Any]]:
    results = list()
    with open(results_path) as file:
        for line in file:
            res = ast.literal_eval(line.strip())
            results.append(res)
    return results


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


def calculate_wer(sequence_a: str, sequence_b: str) -> float:
    # taken from https://github.com/SeanNaren/deepspeech.pytorch/blob/master/decoder.py
    b = set(sequence_a.split() + sequence_b.split())
    word2char = dict(zip(b, range(len(b))))

    w1 = [chr(word2char[w]) for w in sequence_a.split()]
    w2 = [chr(word2char[w]) for w in sequence_b.split()]

    return lvs.distance(''.join(w1), ''.join(w2))


def calculate_bleu2(reference: str, hypothesis: str) -> float:
    return sentence_bleu(
        [reference.split()],
        hypothesis.split(),
        weights=(0.5, 0.5),
        smoothing_function=SmoothingFunction().method0
    )
