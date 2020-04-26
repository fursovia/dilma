import functools

import torch
from allennlp.models import Model
import Levenshtein as lvs


def load_weights(model: Model, path: str, location: str = 'cpu') -> None:
    with open(path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=location))


@functools.lru_cache(maxsize=500)
def calculate_wer(sequence_a: str, sequence_b: str) -> float:
    # taken from https://github.com/SeanNaren/deepspeech.pytorch/blob/master/decoder.py
    b = set(sequence_a.split() + sequence_b.split())
    word2char = dict(zip(b, range(len(b))))

    w1 = [chr(word2char[w]) for w in sequence_a.split()]
    w2 = [chr(word2char[w]) for w in sequence_b.split()]

    return lvs.distance(''.join(w1), ''.join(w2))


def calculate_normalized_wer(sequence_a: str, sequence_b: str) -> float:
    wer = calculate_wer(sequence_a, sequence_b)
    return wer / max(len(sequence_a.split()), len(sequence_b.split()))
