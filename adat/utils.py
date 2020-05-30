import functools
from tqdm import tqdm
from multiprocessing import Pool
from typing import Sequence, Dict, Any, List
import json
import re
import random
from IPython.core.display import display, HTML

import torch
import numpy as np
from allennlp.models import Model
import Levenshtein as lvs


def load_weights(model: Model, path: str, location: str = 'cpu') -> None:
    with open(path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=location))


def load_jsonlines(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path) as file:
        for line in file.readlines():
            data.append(json.loads(line))
    return data


@functools.lru_cache(maxsize=5000)
def calculate_wer(sequence_a: str, sequence_b: str) -> int:
    # taken from https://github.com/SeanNaren/deepspeech.pytorch/blob/master/decoder.py
    b = set(sequence_a.split() + sequence_b.split())
    word2char = dict(zip(b, range(len(b))))

    w1 = [chr(word2char[w]) for w in sequence_a.split()]
    w2 = [chr(word2char[w]) for w in sequence_b.split()]

    return lvs.distance(''.join(w1), ''.join(w2))


def calculate_normalized_wer(sequence_a: str, sequence_b: str) -> float:
    wer = calculate_wer(sequence_a, sequence_b)
    return wer / max(len(sequence_a.split()), len(sequence_b.split()))


def _wer_one_vs_all(x):
    a, bs = x
    return [calculate_wer(a, b) for b in bs]


def pairwise_wer(
    sequences_a: Sequence[str], sequences_b: Sequence[str], n_jobs: int = 5, verbose: bool = False
) -> np.ndarray:
    bar = tqdm if verbose else lambda iterable, total, desc: iterable

    with Pool(n_jobs) as pool:
        distances = list(
            bar(
                pool.imap(_wer_one_vs_all, zip(sequences_a, [sequences_b for _ in sequences_a])),
                total=len(sequences_a),
                desc="# WER {}x{}".format(len(sequences_a), len(sequences_b)),
            )
        )
    return np.array(distances)


def visualize_simple_diff(seq_a: str, seq_b: str, window: int = 3) -> None:

    def _colorize(token: str, color: str) -> str:
        return f"<font color='{color}'>{token}</font>"

    seq_a = seq_a.split()
    seq_b = seq_b.split()

    seq_a_mofidied = []
    for i, token in enumerate(seq_a):
        if token not in seq_b[max(0, i - window):i + window + 1]:
            seq_a_mofidied.append(_colorize(token, "red"))
        else:
            seq_a_mofidied.append(token)

    seq_b_mofidied = []
    for i, token in enumerate(seq_b):
        if token not in seq_a[max(0, i - window):i + window + 1]:
            seq_b_mofidied.append(_colorize(token, "green"))
        else:
            seq_b_mofidied.append(token)

    seq_a_mofidied = " ".join(seq_a_mofidied)
    seq_b_mofidied = " ".join(seq_b_mofidied)

    output = f"{seq_a_mofidied} <br><br> {seq_b_mofidied}"
    display(HTML(output))


def clean_sequence(sequence: str) -> str:
    sequence = sequence.lower()
    sequence = re.sub(r"[^a-zA-Z ]", "", sequence)
    sequence = re.sub(r"\s\s+", " ", sequence).strip()
    return sequence


class SequenceModifier:

    def __init__(
            self,
            vocab: List[str],
            remove_prob: float = 0.05,
            add_prob: float = 0.05,
            replace_prob: float = 0.1
    ) -> None:
        assert sum([remove_prob, add_prob, replace_prob]) > 0.0
        self.vocab = vocab
        self.remove_prob = remove_prob
        self.add_prob = add_prob
        self.replace_prob = replace_prob

    def remove_token(self, sequence: List[str]) -> List[str]:
        samples = np.random.binomial(n=1, p=self.remove_prob, size=len(sequence))
        sequence = [token for i, token in enumerate(sequence) if not samples[i]]
        return sequence

    def replace_token(self, sequence: List[str]) -> List[str]:
        samples = np.random.binomial(n=1, p=self.replace_prob, size=len(sequence))
        new_sequence = [random.choice(self.vocab) if samples[i] else sequence[i] for i in range(len(sequence))]
        return new_sequence

    def add_token(self, sequence: List[str]) -> List[str]:
        new_sequence = sequence + [
            random.choice(self.vocab)
            for _ in range(np.random.binomial(len(sequence), self.add_prob))
        ]
        return new_sequence

    def __call__(self, sequence: str) -> str:
        splitted_sequence = sequence.split()
        if len(splitted_sequence) > 1 and self.remove_prob:
            splitted_sequence = self.remove_token(splitted_sequence)

        if self.replace_prob:
            splitted_sequence = self.replace_token(splitted_sequence)

        if self.add_prob:
            splitted_sequence = self.add_token(splitted_sequence)
        return " ".join(splitted_sequence)


def normalized_accuracy_drop(
        wers: List[int],
        y_true: List[int],
        y_adv: List[int],
        gamma: float = 1.0
) -> float:
    assert len(y_true) == len(y_adv)
    nads = []
    for wer, lab, alab in zip(wers, y_true, y_adv):
        if wer > 0 and lab != alab:
            nads.append(1 / wer ** gamma)
        else:
            nads.append(0.0)

    return sum(nads) / len(nads)


def normalized_accuracy_drop_with_perplexity(
        wers: List[int],
        y_true: List[int],
        y_adv: List[int],
        perp_true: List[float],
        perp_adv: List[float],
        gamma: float = 1.0
) -> float:
    assert len(y_true) == len(y_adv)
    nads = []
    for wer, lab, alab, pt, pa in zip(wers, y_true, y_adv, perp_true, perp_adv):
        if wer > 0 and lab != alab:
            nads.append((1 / wer ** gamma) * (pt / max(pt, pa)))
        else:
            nads.append(0.0)

    return sum(nads) / len(nads)
