import ast
import functools
import json
from typing import List, Dict, Any

import torch
import numpy as np
from allennlp.models import Model
from allennlp.models import LanguageModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import Levenshtein as lvs


from adat.dataset import LanguageModelingReader


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


def calculate_perplexity(sequences: List[str], model: LanguageModel, reader: LanguageModelingReader) -> List[float]:
    perplexities = []
    for sequence in sequences:
        instance = reader.text_to_instance(sequence)
        perplexity = np.exp(model.forward_on_instance(instance)['loss'])
        perplexities.append(float(perplexity))
    return perplexities


@functools.lru_cache(maxsize=500)
def calculate_wer(sequence_a: str, sequence_b: str) -> float:
    # taken from https://github.com/SeanNaren/deepspeech.pytorch/blob/master/decoder.py
    b = set(sequence_a.split() + sequence_b.split())
    word2char = dict(zip(b, range(len(b))))

    w1 = [chr(word2char[w]) for w in sequence_a.split()]
    w2 = [chr(word2char[w]) for w in sequence_b.split()]

    return lvs.distance(''.join(w1), ''.join(w2))


def calculate_normalized_wer(sequence_a: str, sequence_b: str) -> float:
    return calculate_wer(sequence_a, sequence_b) / max(len(sequence_a.split()), len(sequence_b.split()))


@functools.lru_cache(maxsize=500)
def calculate_bleu2(reference: str, hypothesis: str) -> float:
    return sentence_bleu(
        [reference.split()],
        hypothesis.split(),
        weights=(0.5, 0.5),
        smoothing_function=SmoothingFunction().method2
    )


def get_args_from_path(path: str):
    arguments_to_parse = ['task', 'num_classes', 'max_decoding_steps']
    with open(path) as file:
        args = json.load(file)
    model_info = {arg: args[arg] for arg in arguments_to_parse}
    return model_info
