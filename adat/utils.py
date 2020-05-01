import functools
from tqdm import tqdm
from multiprocessing import Pool
from typing import Sequence, Dict, Any, List
import json
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


def visualize_simple_diff(seq_a: str, seq_b: str) -> None:

    def _colorize(token: str, color: str) -> str:
        return f"<font color='{color}'>{token}</font>"

    seq_a = seq_a.split()
    seq_b = seq_b.split()

    seq_a_mofidied = []
    for token in seq_a:
        if token not in seq_b:
            seq_a_mofidied.append(_colorize(token, "red"))
        else:
            seq_a_mofidied.append(token)

    seq_a_mofidied = " ".join(seq_a_mofidied)

    seq_b_mofidied = []
    for token in seq_b:
        if token not in seq_a:
            seq_b_mofidied.append(_colorize(token, "green"))
        else:
            seq_b_mofidied.append(token)
    seq_b_mofidied = " ".join(seq_b_mofidied)

    output = f"{seq_a_mofidied} <br><br> {seq_b_mofidied}"
    display(HTML(output))
