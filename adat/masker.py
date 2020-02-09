from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from enum import Enum

import numpy as np


MASK_TOKEN = "@MASK@"


class Masker(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def mask(self, sequence: str) -> Tuple[str, List[str]]:
        pass


class Sequential(Masker):
    def __init__(self, maskers: List[Masker]) -> None:
        self.maskers = maskers

    @property
    def name(self):
        return self.__class__.__name__

    def mask(self, sequence: str) -> Tuple[str, List[str]]:
        maskers_applied = []
        for masker in self.maskers:
            sequence, maskers_applied_curr = masker.mask(sequence)
            maskers_applied.extend(maskers_applied_curr)
        return sequence, maskers_applied


class Multiple(Masker):
    def __init__(self, maskers: List[Masker], num_maskers: Optional[int] = None) -> None:
        self.maskers = maskers
        self.num_maskers = num_maskers

    @property
    def name(self):
        return self.__class__.__name__

    def mask(self, sequence: str) -> Tuple[str, List[str]]:
        maskers_applied = []
        num_muskers = self.num_maskers or np.random.binomial(n=len(self.maskers), p=0.25)
        masks_to_apply = sorted(np.random.randint(0, len(self.maskers), size=num_muskers))
        for i in masks_to_apply:
            sequence, maskers_applied_curr = self.maskers[i].mask(sequence)
            maskers_applied.extend(maskers_applied_curr)
        return sequence, maskers_applied


class MultipleWithProbs(Masker):
    def __init__(self, maskers: List[Masker], probs: np.ndarray, num_maskers: Optional[int] = None) -> None:
        assert len(maskers) == len(probs)
        self.maskers = maskers
        self.probs = probs
        self.num_maskers = num_maskers

    @property
    def name(self):
        return self.__class__.__name__

    def mask(self, sequence: str) -> Tuple[str, List[str]]:
        maskers_applied = []
        num_muskers = self.num_maskers or np.random.binomial(n=len(self.maskers), p=0.25)
        maskers = np.random.choice(self.maskers, size=num_muskers, replace=True, p=self.probs)
        for masker in maskers:
            sequence, maskers_applied_curr = masker.mask(sequence)
            maskers_applied.extend(maskers_applied_curr)
        return sequence, maskers_applied


class MaskMasker(Masker):
    """
    Each token in a sequence will be replaced by the MASK_TOKEN with probability `p`
    """
    def __init__(self, p: float = 0.1) -> None:
        self.prob = p

    @property
    def name(self):
        return self.__class__.__name__

    def mask(self, sequence: str) -> Tuple[str, List[str]]:
        splitted_sequence = sequence.split()
        seq_size = len(splitted_sequence)
        samples = np.random.binomial(n=1, p=self.prob, size=seq_size)
        masked_sequence = [MASK_TOKEN if samples[i] else token for i, token in enumerate(splitted_sequence)]
        maskers_applied = [self.name for sample in samples if sample]
        return ' '.join(masked_sequence), maskers_applied


class RemoveMasker(Masker):
    """
    Each token in a sequence will be removed with probability `p`
    """
    def __init__(self, p: float = 0.1) -> None:
        self.prob = p

    @property
    def name(self):
        return self.__class__.__name__

    def mask(self, sequence: str) -> Tuple[str, List[str]]:
        splitted_sequence = sequence.split()
        seq_size = len(splitted_sequence)
        samples = np.random.binomial(n=1, p=self.prob, size=seq_size)
        masked_sequence = [token for i, token in enumerate(splitted_sequence) if not samples[i]]
        maskers_applied = [self.name for sample in samples if sample]
        return ' '.join(masked_sequence), maskers_applied


class AddMaskMasker(Masker):
    """
    Each token in a sequence will be followed with the MASK_TOKEN with probability `p`
    """
    def __init__(self, p: float = 0.1) -> None:
        self.prob = p

    @property
    def name(self):
        return self.__class__.__name__

    def mask(self, sequence: str) -> Tuple[str, List[str]]:
        maskers_applied = []
        splitted_sequence = sequence.split()
        seq_size = len(splitted_sequence)
        samples = np.random.binomial(n=1, p=self.prob, size=seq_size)

        masked_sequence = []
        for i, token in enumerate(splitted_sequence):
            masked_sequence.append(token)
            if samples[i]:
                masked_sequence.append(MASK_TOKEN)
                maskers_applied.append(self.name)
        return ' '.join(masked_sequence), maskers_applied


class DoubleMasker(Masker):
    """
    Each token in a sequence will be doubled with probability `p`
    """
    def __init__(self, p: float = 0.1) -> None:
        self.prob = p

    @property
    def name(self):
        return self.__class__.__name__

    def mask(self, sequence: str) -> Tuple[str, List[str]]:
        maskers_applied = []
        splitted_sequence = sequence.split()
        seq_size = len(splitted_sequence)
        samples = np.random.binomial(n=1, p=self.prob, size=seq_size)

        masked_sequence = []
        for i, token in enumerate(splitted_sequence):
            masked_sequence.append(token)
            if samples[i]:
                masked_sequence.append(token)
                maskers_applied.append(self.name)
        return ' '.join(masked_sequence), maskers_applied


class TwoSwapMasker(Masker):
    def __init__(self, p: float = 0.1) -> None:
        self.prob = p

    @property
    def name(self):
        return self.__class__.__name__

    def mask(self, sequence: str) -> Tuple[str, List[str]]:
        maskers_applied = []
        splitted_sequence = sequence.split()
        seq_size = len(splitted_sequence)

        if seq_size >= 2 and self.prob > np.random.rand():
            i, j = sorted(np.random.choice(seq_size, size=2, replace=False))
            splitted_sequence[i], splitted_sequence[j] = splitted_sequence[j], splitted_sequence[i]
            maskers_applied.append(self.name)
        return ' '.join(splitted_sequence), maskers_applied


class ReplaceMasker(Masker):
    def __init__(self, p: float = 0.1) -> None:
        self.vocab = set()
        self.prob = p

    @property
    def name(self):
        return self.__class__.__name__

    def mask(self, sequence: str) -> Tuple[str, List[str]]:
        maskers_applied = []
        splitted_sequence = sequence.split()
        self.vocab.update(set(splitted_sequence))
        seq_size = len(splitted_sequence)
        samples = np.random.binomial(n=1, p=self.prob, size=seq_size)
        new_sequence = []
        for i, token in enumerate(splitted_sequence):
            if samples[i]:
                random_token = np.random.choice(list(self.vocab))
                new_sequence.append(random_token)
                maskers_applied.append(self.name)
            else:
                new_sequence.append(token)

        return ' '.join(new_sequence), maskers_applied


class AddStrategy(str, Enum):
    END = 'end'
    START = 'start'
    RANDOM = 'random'


class AddMasker(Masker):
    _max_num_to_add = 5

    def __init__(self, p: float = 0.1, strategy: AddStrategy = AddStrategy.RANDOM) -> None:
        self.vocab = set()
        self.prob = p
        self.strategy = strategy

    @property
    def name(self):
        return self.__class__.__name__ + self.strategy.value

    def mask(self, sequence: str) -> Tuple[str, List[str]]:
        maskers_applied = []
        splitted_sequence = sequence.split()
        self.vocab.update(set(splitted_sequence))
        seq_size = len(splitted_sequence)

        new_sequence = []
        if self.strategy == AddStrategy.RANDOM:
            samples = np.random.binomial(n=1, p=self.prob, size=seq_size)
            for i, token in enumerate(splitted_sequence):
                new_sequence.append(token)
                if samples[i]:
                    random_token = np.random.choice(list(self.vocab))
                    new_sequence.append(random_token)
                    maskers_applied.append(self.name)
        elif self.strategy == AddStrategy.END:
            new_sequence.extend(splitted_sequence)
            num_tokens_to_add = np.random.binomial(n=self._max_num_to_add, p=self.prob)
            for _ in range(num_tokens_to_add):
                random_token = np.random.choice(list(self.vocab))
                new_sequence.append(random_token)
                maskers_applied.append(self.name)
        elif self.strategy == AddStrategy.START:
            num_tokens_to_add = np.random.binomial(n=self._max_num_to_add, p=self.prob)
            for _ in range(num_tokens_to_add):
                random_token = np.random.choice(list(self.vocab))
                new_sequence.append(random_token)
                maskers_applied.append(self.name)
            new_sequence.extend(splitted_sequence)
        else:
            raise NotImplementedError

        return ' '.join(new_sequence), maskers_applied


def get_default_masker() -> MultipleWithProbs:
    """
    Random swaps, replacements, additions and deletions
    """
    mask = MultipleWithProbs(
        [
            TwoSwapMasker(),
            RemoveMasker(),
            DoubleMasker(),
            ReplaceMasker(),
            AddMasker(strategy=AddStrategy.RANDOM),
            AddMasker(strategy=AddStrategy.END),
            AddMasker(strategy=AddStrategy.START),
        ],
        probs=np.array([0.3, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1]),
        num_maskers=None  # 1.75 in average
    )
    return mask


def get_default_masker_names() -> List[str]:
    names = []
    masker = get_default_masker()
    for m in masker.maskers:
        names.append(m.name)
    return names
