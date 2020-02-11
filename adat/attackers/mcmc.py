from abc import ABC, abstractmethod
from typing import List, Dict
from functools import lru_cache
import random

import torch
import numpy as np
from torch.distributions import Normal
from allennlp.models.basic_classifier import BasicClassifier
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import move_to_device
from allennlp.data.dataset import Batch

from adat.attackers.attacker import SamplerOutput
from adat.utils import calculate_normalized_wer
from adat.models import OneLanguageSeq2SeqModel


PROB_DIFF = -100


class Proposal(ABC):
    @abstractmethod
    def sample(self, curr_state: torch.Tensor) -> torch.Tensor:
        pass


class NormalProposal(Proposal):
    def __init__(self, variance: float = 0.1) -> None:
        self.variance = variance

    def sample(self, curr_state: torch.Tensor) -> torch.Tensor:
        return Normal(curr_state, torch.ones_like(curr_state) * (self.variance ** 0.5)).sample()


# TODO: num_steps passed
# TODO: take last if self.history is None (self.full_history)
class Sampler(ABC):
    def __init__(
            self,
            proposal_distribution: Proposal,
            classification_model: BasicClassifier,
            classification_reader: DatasetReader,
            generation_model: OneLanguageSeq2SeqModel,
            generation_reader: DatasetReader,
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

        self.label_prob_to_drop = None
        self.initial_sequence = None
        self.current_state = None
        self.initial_prob = None
        self.history: List[SamplerOutput] = []

    def set_label_to_attack(self, label: int = 1) -> None:
        self.label_prob_to_drop = label

    def set_input(self, initial_sequence: str) -> None:
        self.initial_sequence = initial_sequence
        with torch.no_grad():
            self.current_state = self.generation_model.get_state_for_beam_search(
                self._seq_to_input(
                    self.initial_sequence,
                    self.generation_reader,
                    self.generation_vocab
                )['source_tokens']
            )

        self.initial_prob = self.predict_prob(self.initial_sequence)
        # print(self.generate_from_state(self.current_state.copy()))

    def empty_history(self) -> None:
        self.history = []

    def _seq_to_input(self, seq: str, reader: DatasetReader,
                      vocab: Vocabulary) -> Dict[str, Dict[str, torch.LongTensor]]:
        instance = reader.text_to_instance(seq)
        batch = Batch([instance])
        batch.index_instances(vocab)
        return move_to_device(batch.as_tensor_dict(), self.device)

    def generate_from_state(self, state: Dict[str, torch.Tensor]) -> str:
        with torch.no_grad():
            pred_output = self.generation_model.beam_search(state)
            return ' '.join(self.generation_model.decode(pred_output)['predicted_tokens'][0])

    @lru_cache(maxsize=1000)
    def predict_prob(self, sequence: str) -> float:
        assert self.label_prob_to_drop is not None, 'You must run `.set_label_to_drop()` first.'
        with torch.no_grad():
            return self.classification_model.forward_on_instance(
                self.classification_reader.text_to_instance(sequence)
            )['probs'][self.label_prob_to_drop]

    @lru_cache(maxsize=1000)
    def get_output(self, generated_sequence: str) -> SamplerOutput:
        new_prob = self.predict_prob(generated_sequence)
        new_wer = calculate_normalized_wer(self.initial_sequence, generated_sequence)
        prob_diff = self.initial_prob - new_prob
        return SamplerOutput(generated_sequence=generated_sequence, wer=new_wer, prob_diff=prob_diff)

    @abstractmethod
    def step(self):
        pass

    def sample(self, num_steps: int = 100) -> List[SamplerOutput]:
        for _ in range(num_steps):
            self.step()
        return self.history

    def sample_until_satisfied(self, max_steps: int = 200, wer: float = 0.2, prob_drop: float = 1.5) -> SamplerOutput:
        """
        Sample until the wer <= X and the probability drops by `prob_drop` times minimum
        """

        for _ in range(max_steps):
            self.step()
            # if self.history:
            #     print(self.initial_prob / (self.initial_prob - self.history[-1].prob_diff),
            #           self.history[-1].wer, self.history[-1].generated_sequence)
            if self.history and self.history[-1].wer <= wer and \
                    (self.initial_prob / (self.initial_prob - self.history[-1].prob_diff) >= prob_drop):
                return self.history[-1]
        # print('did not found', len(self.history))
        return SamplerOutput(generated_sequence=self.initial_sequence) if not self.history \
            else max(self.history, key=lambda x: x.prob_diff)


# TODO: should I update state?
class RandomSampler(Sampler):
    def step(self) -> None:
        assert self.current_state is not None, 'Run `set_input()` first'
        new_state = self.current_state.copy()
        new_state['decoder_hidden'] = self.proposal_distribution.sample(new_state['decoder_hidden'])
        generated_seq = self.generate_from_state(new_state.copy())

        if generated_seq:
            self.current_state = new_state
            output = self.get_output(generated_seq)
            self.history.append(output)


class MCMCSampler(Sampler):
    def __init__(
            self,
            proposal_distribution: Proposal,
            classification_model: BasicClassifier,
            classification_reader: DatasetReader,
            generation_model: OneLanguageSeq2SeqModel,
            generation_reader: DatasetReader,
            sigma_class: float = 1.0,
            sigma_wer: float = 0.5,
            device: int = -1
    ) -> None:
        super().__init__(proposal_distribution, classification_model, classification_reader,
                         generation_model, generation_reader, device)
        self.sigma_class = sigma_class
        self.sigma_wer = sigma_wer

    def step(self) -> None:
        assert self.current_state is not None, 'Run `set_input()` first'
        new_state = self.current_state.copy()
        new_state['decoder_hidden'] = self.proposal_distribution.sample(new_state['decoder_hidden'])
        generated_seq = self.generate_from_state(new_state.copy())

        if generated_seq:
            output = self.get_output(generated_seq)

            prob_diff = output.prob_diff if output.prob_diff > 0 else PROB_DIFF
            exp_base = (-output.wer / self.sigma_wer) + (-1 + prob_diff) / self.sigma_class

            acceptance_probability = min(
                [
                    1.,
                    np.exp(exp_base)
                ]
            )

            if acceptance_probability > random.random():
                self.current_state = new_state
                self.history.append(output)
