from abc import ABC, abstractmethod
from typing import List

from dataclasses import dataclass


@dataclass
class AttackerOutput:
    generated_sequence: str
    label: int
    wer: float = None
    prob_diff: float = None
    acceptance_probability: float = None


def find_best_output(outputs: List[AttackerOutput], initial_label: int) -> AttackerOutput:
    changed_label_outputs = []
    for output in outputs:
        if output.label != initial_label:
            changed_label_outputs.append(output)

    if changed_label_outputs:
        best_output = min(changed_label_outputs, key=lambda x: x.wer)
    else:
        best_output = max(outputs, key=lambda x: x.prob_diff)

    return best_output
