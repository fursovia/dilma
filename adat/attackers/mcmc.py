from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
import random

import torch
import numpy as np
from torch.distributions import Normal
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import move_to_device
from allennlp.data.dataset import Batch

from adat.attackers.attacker import AttackerOutput, find_best_output
from adat.utils import calculate_normalized_wer
from adat.models import MaskedCopyNet, Classifier
from adat.dataset import ClassificationReader, CopyNetReader


PROB_DIFF = -100


class Proposal(ABC):
    @abstractmethod
    def sample(self, curr_state: torch.Tensor) -> torch.Tensor:
        pass


class NormalProposal(Proposal):
    def __init__(self, scale: float = 0.1) -> None:
        self.scale = scale  # standard deviation

    def sample(self, curr_state: torch.Tensor) -> torch.Tensor:
        return Normal(curr_state, torch.ones_like(curr_state) * self.scale).sample()


class Sampler(ABC):
    def __init__(
            self,
            proposal_distribution: Proposal,
            classification_model: Classifier,
            classification_reader: ClassificationReader,
            generation_model: MaskedCopyNet,
            generation_reader: CopyNetReader,
            device: int = -1
    ) -> None:
        self.proposal_distribution = proposal_distribution

        # models
        self.classification_model = classification_model
        self.classification_model.eval()
        self.classification_reader = classification_reader
        self.classification_vocab = self.classification_model.vocab

        self.generation_model = generation_model
        self.generation_model.eval()
        self.generation_reader = generation_reader
        self.generation_vocab = self.generation_model.vocab

        self.device = device
        if self.device >= 0 and torch.cuda.is_available():
            self.classification_model.cuda(self.device)
            self.generation_model.cuda(self.device)
        else:
            self.classification_model.cpu()
            self.generation_model.cpu()

        self.label_to_attack = None
        self.initial_sequence = None
        self.current_state = None
        self.initial_prob = None
        self.history: List[AttackerOutput] = []

    def set_label_to_attack(self, label: int = 1) -> None:
        self.label_to_attack = label

    def set_input(self, initial_sequence: str, mask_tokens: Optional[List[str]] = None) -> None:
        self.initial_sequence = initial_sequence
        with torch.no_grad():
            inputs = self._seq_to_input(
                self.initial_sequence,
                self.generation_reader,
                self.generation_vocab,
                mask_tokens=mask_tokens
            )
            self.current_state = self.generation_model.encode(
                source_tokens=inputs['source_tokens'],
                mask_tokens=inputs['mask_tokens']
            )

        self.current_state = self.generation_model.init_decoder_state(self.current_state)
        self.initial_prob, _ = self.predict_prob_and_label(self.initial_sequence)

    def empty_history(self) -> None:
        self.history = []
        self.label_to_attack = None
        self.initial_sequence = None
        self.current_state = None
        self.initial_prob = None

    def _seq_to_input(
            self,
            seq: str,
            reader: CopyNetReader,
            vocab: Vocabulary,
            mask_tokens: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, torch.LongTensor]]:
        instance = reader.text_to_instance(seq, maskers_applied=mask_tokens)
        batch = Batch([instance])
        batch.index_instances(vocab)
        return move_to_device(batch.as_tensor_dict(), self.device)

    def generate_from_state(self, state: Dict[str, torch.Tensor]) -> List[str]:
        with torch.no_grad():
            state = self.generation_model.init_decoder_state(state)
            pred_output = self.generation_model.beam_search(state)
            predicted_sequences = []
            for seq in self.generation_model.decode(pred_output)['predicted_tokens'][0]:
                predicted_sequences.append(' '.join(seq))
            return predicted_sequences

    @lru_cache(maxsize=1000)
    def predict_prob_and_label(self, sequence: str) -> Tuple[float, int]:
        assert self.label_to_attack is not None, 'You must run `.set_label_to_attack()` first.'
        # logits, probs, label
        with torch.no_grad():
            predictions = self.classification_model.forward_on_instance(
                self.classification_reader.text_to_instance(sequence)
            )
            prob = predictions['probs'][self.label_to_attack]
            label = predictions['label']
            return float(prob), int(label)

    @lru_cache(maxsize=1000)
    def get_output(self, generated_sequence: str) -> AttackerOutput:
        new_prob, new_label = self.predict_prob_and_label(generated_sequence)
        new_wer = calculate_normalized_wer(self.initial_sequence, generated_sequence)
        prob_diff = self.initial_prob - new_prob
        return AttackerOutput(generated_sequence=generated_sequence, label=new_label, wer=new_wer, prob_diff=prob_diff)

    @abstractmethod
    def step(self):
        pass

    def sample_until_label_is_changed(self, max_steps: int = 200, early_stopping: bool = False) -> AttackerOutput:

        for _ in range(max_steps):
            self.step()
            if early_stopping and self.history and self.history[-1].label != self.label_to_attack:
                return self.history[-1]

        if self.history:
            output = find_best_output(self.history, self.label_to_attack)
        else:
            output = AttackerOutput(generated_sequence=self.initial_sequence, label=self.label_to_attack)

        return output


# TODO: should I update state?
class RandomSampler(Sampler):
    def step(self) -> None:
        assert self.current_state is not None, 'Run `set_input()` first'
        new_state = self.current_state.copy()
        new_state['decoder_hidden'] = self.proposal_distribution.sample(new_state['decoder_hidden'])
        generated_sequences = self.generate_from_state(new_state.copy())

        curr_outputs = list()
        # we generated `beam_size` adversarial examples
        for generated_seq in generated_sequences:
            # sometimes len(generated_seq) = 0
            if generated_seq:
                curr_outputs.append(self.get_output(generated_seq))

        if curr_outputs:
            output = find_best_output(curr_outputs, self.label_to_attack)
            self.current_state = new_state
            self.history.append(output)


class MCMCSampler(Sampler):
    def __init__(
            self,
            proposal_distribution: Proposal,
            classification_model: Classifier,
            classification_reader: ClassificationReader,
            generation_model: MaskedCopyNet,
            generation_reader: CopyNetReader,
            sigma_class: float = 1.0,
            sigma_wer: float = 0.5,
            device: int = -1
    ) -> None:
        super().__init__(
            proposal_distribution=proposal_distribution,
            classification_model=classification_model,
            classification_reader=classification_reader,
            generation_model=generation_model,
            generation_reader=generation_reader,
            device=device
        )
        self.sigma_class = sigma_class
        self.sigma_wer = sigma_wer

    def step(self) -> None:
        assert self.current_state is not None, 'Run `set_input()` first'
        new_state = self.current_state.copy()
        new_state['decoder_hidden'] = self.proposal_distribution.sample(new_state['decoder_hidden'])
        generated_sequences = self.generate_from_state(new_state.copy())

        curr_outputs = list()
        # we generated `beam_size` adversarial examples
        for generated_seq in generated_sequences:
            # sometimes len(generated_seq) = 0
            if generated_seq:
                curr_outputs.append(self.get_output(generated_seq))

        if curr_outputs:
            output = find_best_output(curr_outputs, self.label_to_attack)
            prob_diff = output.prob_diff if output.prob_diff > 0 else PROB_DIFF
            exp_base = (-output.wer / self.sigma_wer) + (-1 + prob_diff) / self.sigma_class

            acceptance_probability = min(
                [
                    1.,
                    np.exp(exp_base)
                ]
            )
            output.acceptance_probability = acceptance_probability
            if acceptance_probability > random.random():
                self.current_state = new_state
                self.history.append(output)
