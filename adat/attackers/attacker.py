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
    attacked_label: int
    adversarial_label: int


class Attacker(ABC):
    @abstractmethod
    def attack(self, sequence_to_attack: str, **kwargs) -> AttackerOutput:
        pass


def find_best_attack(outputs: List[AttackerOutput]) -> AttackerOutput:
    changed_label_outputs = []
    for output in outputs:
        if output.attacked_label != output.adversarial_label:
            changed_label_outputs.append(output)

    if changed_label_outputs:
        sorted_outputs = sorted(changed_label_outputs, key=lambda x: x.prob_diff, reverse=True)
        best_output = min(sorted_outputs, key=lambda x: x.wer)
    else:
        best_output = max(outputs, key=lambda x: x.prob_diff)

    return best_output
