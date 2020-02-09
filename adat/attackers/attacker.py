from abc import ABC, abstractmethod

import numpy as np
from dataclasses import dataclass


class Attacker(ABC):
    @abstractmethod
    def attack(self, **kwargs):
        pass


@dataclass
class SamplerOutput:
    generated_sequence: str
    prob: float = 1.0
    bleu: float = 0.0
    prob_diff: float = 0.0
    prob_drop: float = 1.0
    bleu_diff: float = -1.0
    bleu_drop: float = np.inf
    acceptance_probability: float = 0.0


@dataclass
class AttackerOutput:
    generated_sequence: str
    prob: float = 1.0
    prob_diff: float = 0.0
    wer: float = 0.0
