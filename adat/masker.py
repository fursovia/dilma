from abc import ABC, abstractmethod
from typing import List

import numpy as np


MASK_TOKEN = "@MASK@"


class Masker(ABC):

    @abstractmethod
    def mask(self, sequence: str) -> str:
        pass


class Sequential(Masker):
    def __init__(self, maskers: List[Masker]) -> None:
        self.maskers = maskers

    def mask(self, sequence: str) -> str:
        for masker in self.maskers:
            sequence = masker.mask(sequence)
        return sequence


class Multiple(Masker):
    def __init__(self, maskers: List[Masker], num_maskers: int = 1) -> None:
        self.maskers = maskers
        self.num_maskers = num_maskers

    def mask(self, sequence: str) -> str:
        masks_to_apply = sorted(np.random.randint(0, len(self.maskers), size=self.num_maskers))
        for i in masks_to_apply:
            sequence = self.maskers[i].mask(sequence)
        return sequence


class MaskMasker(Masker):
    """
    Each token in a sequence will be replaced by the MASK_TOKEN with probability `p`
    """
    def __init__(self, p: float = 0.1) -> None:
        self.prob = p

    def mask(self, sequence: str) -> str:
        splitted_sequence = sequence.split()
        seq_size = len(splitted_sequence)
        samples = np.random.binomial(n=1, p=self.prob, size=seq_size)
        masked_sequence = [MASK_TOKEN if samples[i] else token for i, token in enumerate(splitted_sequence)]
        return ' '.join(masked_sequence)


class RemoveMasker(Masker):
    """
    Each token in a sequence will be removed with probability `p`
    """
    def __init__(self, p: float = 0.1) -> None:
        self.prob = p

    def mask(self, sequence: str) -> str:
        splitted_sequence = sequence.split()
        seq_size = len(splitted_sequence)
        samples = np.random.binomial(n=1, p=self.prob, size=seq_size)
        masked_sequence = [token for i, token in enumerate(splitted_sequence) if not samples[i]]
        return ' '.join(masked_sequence)


class AddMasker(Masker):
    """
    Each token in a sequence will be followed with the MASK_TOKEN with probability `p`
    """
    def __init__(self, p: float = 0.1) -> None:
        self.prob = p

    def mask(self, sequence: str) -> str:
        splitted_sequence = sequence.split()
        seq_size = len(splitted_sequence)
        samples = np.random.binomial(n=1, p=self.prob, size=seq_size)

        masked_sequence = []
        for i, token in enumerate(splitted_sequence):
            masked_sequence.append(token)
            if samples[i]:
                masked_sequence.append(MASK_TOKEN)
        return ' '.join(masked_sequence)


class DoubleMasker(Masker):
    """
    Each token in a sequence will be doubled with probability `p`
    """
    def __init__(self, p: float = 0.1) -> None:
        self.prob = p

    def mask(self, sequence: str) -> str:
        splitted_sequence = sequence.split()
        seq_size = len(splitted_sequence)
        samples = np.random.binomial(n=1, p=self.prob, size=seq_size)

        masked_sequence = []
        for i, token in enumerate(splitted_sequence):
            masked_sequence.append(token)
            if samples[i]:
                masked_sequence.append(token)
        return ' '.join(masked_sequence)


class TwoSwapMasker(Masker):
    def __init__(self, p: float = 0.1) -> None:
        self.prob = p

    def mask(self, sequence: str) -> str:
        splitted_sequence = sequence.split()
        seq_size = len(splitted_sequence)

        if seq_size >= 2 and self.prob > np.random.rand():
            i, j = sorted(np.random.choice(seq_size, size=2, replace=False))
            splitted_sequence[i], splitted_sequence[j] = splitted_sequence[j], splitted_sequence[i]
        return ' '.join(splitted_sequence)
