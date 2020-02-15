from abc import ABC, abstractmethod

from dataclasses import dataclass


class Attacker(ABC):
    @abstractmethod
    def attack(self, **kwargs):
        pass


@dataclass
class SamplerOutput:
    generated_sequence: str
    label: int
    wer: float = 0.0
    prob_diff: float = 0.0
    acceptance_probability: float = 1.0


@dataclass
class AttackerOutput:
    generated_sequence: str
    prob: float = 1.0
    prob_diff: float = 0.0
    wer: float = 0.0
