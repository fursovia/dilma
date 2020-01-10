from abc import ABC, abstractmethod
from typing import List, Dict
from copy import deepcopy

import torch
import numpy as np
from torch.distributions import Normal
from allennlp.models.basic_classifier import BasicClassifier
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import move_to_device
from allennlp.data.dataset import Batch

from adat.utils import calculate_bleu2
from adat.models import OneLanguageSeq2SeqModel


class Proposal(ABC):

    @abstractmethod
    def sample(self, curr_state: torch.Tensor) -> torch.Tensor:
        pass


class NormalProposal(Proposal):
    def __init__(self, variance: float = 0.1) -> None:
        self.variance = variance

    def sample(self, curr_state: torch.Tensor) -> torch.Tensor:
        return Normal(curr_state, torch.ones_like(curr_state) * (self.variance ** 0.5)).sample()


class MCMCSampler:
    def __init__(
            self,
            proposal_distribution: Proposal,
            classification_model: BasicClassifier,
            classification_reader: DatasetReader,
            generation_model: OneLanguageSeq2SeqModel,
            generation_reader: DatasetReader,
            initial_sequence: str,
            bleu: bool = True,
            l2_norm: bool = True,
            sigma: float = 0.8,
            device: int = -1
    ) -> None:
        self.proposal_distribution = proposal_distribution

        # models
        self.classification_model = classification_model
        self.classification_model.training = False
        self.classification_reader = classification_reader
        self.classification_vocab = self.classification_model.vocab

        self.generation_model = generation_model
        self.generation_model.training = False
        self.generation_reader = generation_reader
        self.generation_vocab = self.generation_model.vocab

        self.device = device
        if self.device >= 0 and torch.cuda.is_available():
            self.classification_model.cuda(self.device)
            self.generation_model.cuda(self.device)
        else:
            self.classification_model.cpu()
            self.generation_model.cpu()

        # initial hidden state
        self.initial_sequence = initial_sequence
        self.current_state = self.generation_model.get_state_for_beam_search(
            self._seq_to_input(self.initial_sequence, self.generation_reader, self.generation_vocab)
        )

        self.bleu = bleu
        self.l2_norm = l2_norm
        self.sigma = sigma

        self.curr_prob = self.predict_prob(self.initial_sequence)
        self.curr_bleu = calculate_bleu2(self.initial_sequence, self.generate_from_state(self.current_state))
        self.history = []

    def _seq_to_input(self, seq: str, reader: DatasetReader, vocab: Vocabulary):
        instance = reader.text_to_instance(seq)
        batch = Batch([instance])
        batch.index_instances(vocab)
        return move_to_device(batch.as_tensor_dict(), self.device)

    def generate_from_state(self, state: Dict[str, torch.Tensor]) -> str:
        with torch.no_grad:
            return ' '.join(self.generation_model.beam_search(state)['predicted_tokens'][0])

    def predict_prob(self, sequence: str) -> float:
        with torch.no_grad:
            return self.classification_model.forward_on_instance(
                self.classification_reader.text_to_instance(sequence)
            )['probs'][1]

    def step(self):
        new_state = deepcopy(self.current_state)
        new_state['decoder_hidden'] = self.proposal_distribution.sample(new_state['decoder_hidden'])
        new_prob = self.predict_prob(new_state)
        generated_sentence = self.generate_from_state(new_state)
        new_bleu = calculate_bleu2(self.initial_sequence, generated_sentence)

        acceptance_probability = min(
            [
                1.,
                np.exp(
                    -(1 - (new_prob - self.curr_prob)) - (1 - (new_bleu - self.curr_bleu))
                    - np.linalg.norm(
                        self.current_state['decoder_hidden'].cpu().numpy() - new_state['decoder_hidden'].cpu().numpy()
                    )
                ) / (2 * self.sigma ** 2)
            ]
        )

        if np.random.rand() < acceptance_probability:
            self.current_state = new_state
            self.curr_prob = new_prob
            self.curr_bleu = new_bleu

            self.history.append(generated_sentence)

    def sample(self, num_steps: int = 100) -> List[torch.Tensor]:
        for _ in range(num_steps):
            self.step()
        return self.history
