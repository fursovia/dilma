from typing import List
from abc import ABC, abstractmethod

from dataclasses import dataclass


@dataclass
class AttackerOutput:
    sequence: str
    probability: float
    adversarial_sequence: str
    adversarial_probability: float
    wer: int
    prob_diff: float


class Attacker(ABC):
    @abstractmethod
    def attack(self, sequence_to_attack: str, **kwargs) -> AttackerOutput:
        pass


def find_best_attack(outputs: List[AttackerOutput], threshold: float = 0.1) -> AttackerOutput:
    dropped_by_threshold_outputs = []
    for output in outputs:
        output.prob_diff = output.probability - output.adversarial_probability
        if output.prob_diff >= threshold:
            dropped_by_threshold_outputs.append(output)

    if dropped_by_threshold_outputs:
        sorted_outputs = sorted(dropped_by_threshold_outputs, key=lambda x: x.prob_diff, reverse=True)
        best_output = min(sorted_outputs, key=lambda x: x.wer)
    else:
        best_output = max(outputs, key=lambda x: x.prob_diff)

    return best_output
