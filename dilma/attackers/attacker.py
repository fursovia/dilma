from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

from dataclasses import dataclass
from dataclasses_json import dataclass_json
from allennlp.common.registrable import Registrable
from allennlp.models import Model
import torch

from dilma.common import SequenceData


@dataclass_json
@dataclass
class AttackerOutput:
    data: SequenceData
    adversarial_data: SequenceData
    probability: float  # original probability
    adversarial_probability: float
    prob_diff: float
    wer: int
    history: Optional[List[Dict[str, Any]]] = None


class Attacker(ABC, Registrable):
    def __init__(
            self,
            substitute_classifier: Model,
            reader: TransactionsDatasetReader,
            device: int = -1,
    ) -> None:
        self.classifier = classifier
        self.classifier.eval()
        self.reader = reader

        self.device = device
        if self.device >= 0 and torch.cuda.is_available():
            self.classifier.cuda(self.device)

    @abstractmethod
    def attack(self, data_to_attack: TransactionsData) -> AttackerOutput:
        pass

    # TODO: add typing
    def get_clf_probs(self, inputs) -> torch.Tensor:
        probs = self.classifier(**inputs)["probs"][0]
        return probs

    def probs_to_label(self, probs: torch.Tensor) -> int:
        label_idx = probs.argmax().item()
        label = self.index_to_label(label_idx)
        return label

    def index_to_label(self, label_idx: int) -> int:
        label = self.classifier.vocab.get_index_to_token_vocabulary("labels").get(label_idx, str(label_idx))
        return int(label)

    def label_to_index(self, label: int) -> int:
        label_idx = self.classifier.vocab.get_token_to_index_vocabulary("labels").get(str(label), label)
        return label_idx

    @staticmethod
    def find_best_attack(outputs: List[AttackerOutput]) -> AttackerOutput:
        if len(outputs) == 1:
            return outputs[0]

        changed_label_outputs = []
        for output in outputs:
            if output.data["label"] != output.adversarial_data["label"] and output.wer > 0:
                changed_label_outputs.append(output)

        if changed_label_outputs:
            sorted_outputs = sorted(changed_label_outputs, key=lambda x: x.prob_diff, reverse=True)
            best_output = min(sorted_outputs, key=lambda x: x.wer)
        else:
            best_output = max(outputs, key=lambda x: x.prob_diff)

        return best_output
