from abc import ABC, abstractmethod

import numpy as np


MASK_TOKEN = "@MASK@"


class Masker(ABC):

    @abstractmethod
    def mask(self, sequence: str) -> str:
        pass


class SimpleMasker(Masker):
    def __init__(self, p: float) -> None:
        self.prob = p

    def mask(self, sequence: str) -> str:
        splitted_sequence = sequence.split()
        text_size = len(splitted_sequence)
        samples = np.random.binomial(n=1, p=self.prob, size=text_size)
        masked_sequence = [MASK_TOKEN if samples[i] else token for i, token in enumerate(splitted_sequence)]
        return ' '.join(masked_sequence)
