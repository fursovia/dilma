from typing import List
from abc import ABC, abstractmethod

from dataclasses import dataclass


@dataclass
class AttackerOutput:
    sequence: str
    label: int
    adversarial_sequence: str
    adversarial_label: int = None
    wer: float = None
    prob_diff: float = None
    acceptance_probability: float = None


def find_best_output(outputs: List[AttackerOutput], initial_label: int) -> AttackerOutput:
    changed_label_outputs = []
    for output in outputs:
        if output.adversarial_label != initial_label:
            changed_label_outputs.append(output)

    if changed_label_outputs:
        sorted_outputs = sorted(changed_label_outputs, key=lambda x: x.prob_diff, reverse=True)
        best_output = min(sorted_outputs, key=lambda x: x.wer)
    else:
        best_output = max(outputs, key=lambda x: x.prob_diff)

    return best_output


class Attacker(ABC):
    def __init__(self) -> None:
        self.label_to_attack = None
        self.initial_sequence = None
        self.current_state = None
        self.initial_prob = None
        self.history: List[AttackerOutput] = list()

    def empty_history(self) -> None:
        self.label_to_attack = None
        self.initial_sequence = None
        self.current_state = None
        self.initial_prob = None
        self.history = list()

    @abstractmethod
    def step(self) -> None:
        pass

    def sample_until_label_is_changed(self, max_steps: int = 200, early_stopping: bool = False) -> AttackerOutput:

        for _ in range(max_steps):
            self.step()
            if early_stopping and self.history and self.history[-1].label != self.label_to_attack:
                return self.history[-1]

        if self.history:
            output = find_best_output(self.history, self.label_to_attack)
        else:
            output = AttackerOutput(
                sequence=self.initial_sequence,
                label=self.label_to_attack,
                adversarial_sequence=self.initial_sequence,
                adversarial_label=self.label_to_attack,
                wer=0.0,
                prob_diff=0.0
            )

        return output

