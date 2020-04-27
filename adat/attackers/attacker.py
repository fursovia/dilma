# from typing import List, Optional, Dict
# from abc import ABC, abstractmethod
#
# import torch
# from dataclasses import dataclass
# # from allennlp.data.dataset import Batch
# from allennlp.nn.util import move_to_device
# from allennlp.data.vocabulary import Vocabulary
#
# from adat.dataset_readers.copynet import CopyNetReader
#
#
# @dataclass
# class AttackerOutput:
#     sequence: str
#     label: int
#     adversarial_sequence: str
#     adversarial_label: int = None
#     wer: float = None
#     prob_diff: float = None
#     acceptance_probability: float = None
#
#
# def find_best_output(outputs: List[AttackerOutput], initial_label: int) -> AttackerOutput:
#     changed_label_outputs = []
#     for output in outputs:
#         if output.adversarial_label != initial_label:
#             changed_label_outputs.append(output)
#
#     if changed_label_outputs:
#         sorted_outputs = sorted(changed_label_outputs, key=lambda x: x.prob_diff, reverse=True)
#         best_output = min(sorted_outputs, key=lambda x: x.wer)
#     else:
#         best_output = max(outputs, key=lambda x: x.prob_diff)
#
#     return best_output
#
#
# def find_best_output(outputs: List[AttackerOutput], initial_label: int, wer_max: int = 5) -> AttackerOutput:
#     outputs_new = [output for output in outputs if output.wer <= wer_max]
#     outputs = outputs_new if outputs_new else outputs
#     changed_label_outputs = []
#     for output in outputs:
#         if output.adversarial_label != initial_label:
#             changed_label_outputs.append(output)
#
#     if changed_label_outputs:
#         sorted_outputs = sorted(changed_label_outputs, key=lambda x: x.prob_diff, reverse=True)
#         best_output = min(sorted_outputs, key=lambda x: x.wer)
#     else:
#         best_output = max(outputs, key=lambda x: x.prob_diff)
#
#     return best_output
#
#
# class Attacker(ABC):
#     def __init__(self, device: int = -1) -> None:
#         self.device = device
#         self.label_to_attack = None
#         self.initial_sequence = None
#         self.current_state = None
#         self.initial_prob = None
#         self.history: List[AttackerOutput] = list()
#
#     def empty_history(self) -> None:
#         self.label_to_attack = None
#         self.initial_sequence = None
#         self.current_state = None
#         self.initial_prob = None
#         self.history = list()
#
#     def set_label_to_attack(self, label: int) -> None:
#         self.label_to_attack = label
#
#     def _sequence2batch(
#             self,
#             sequence: str,
#             reader: CopyNetReader,
#             vocab: Vocabulary,
#             mask_tokens: Optional[List[str]] = None
#     ) -> Dict[str, Dict[str, torch.LongTensor]]:
#         instance = reader.text_to_instance(sequence, maskers_applied=mask_tokens)
#         batch = Batch([instance])
#         batch.index_instances(vocab)
#         return move_to_device(batch.as_tensor_dict(), self.device)
#
#     @abstractmethod
#     def step(self) -> None:
#         pass
#
#     def sample_until_label_is_changed(self, max_steps: int = 200, early_stopping: bool = False) -> AttackerOutput:
#
#         for _ in range(max_steps):
#             self.step()
#             if early_stopping and self.history and self.history[-1].label != self.label_to_attack:
#                 return self.history[-1]
#
#         if self.history:
#             output = find_best_output(self.history, self.label_to_attack)
#         else:
#             output = AttackerOutput(
#                 sequence=self.initial_sequence,
#                 label=self.label_to_attack,
#                 adversarial_sequence=self.initial_sequence,
#                 adversarial_label=self.label_to_attack,
#                 wer=0.0,
#                 prob_diff=0.0
#             )
#
#         return output
#
