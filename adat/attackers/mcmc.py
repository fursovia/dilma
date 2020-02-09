from abc import ABC, abstractmethod
from typing import List, Dict

import torch
import numpy as np
from torch.distributions import Normal
from allennlp.models.basic_classifier import BasicClassifier
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import move_to_device
from allennlp.data.dataset import Batch

from adat.attackers.attacker import SamplerOutput
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
        self.initial_bleu = None
        self.history: List[SamplerOutput] = []

    def set_label_to_drop(self, label: int = 1) -> None:
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
        self.initial_bleu = calculate_bleu2(self.initial_sequence, self.generate_from_state(self.current_state.copy()))

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

    def predict_prob(self, sequence: str) -> float:
        assert self.label_prob_to_drop is not None, 'You must run `.set_label_to_drop()` first.'
        with torch.no_grad():
            return self.classification_model.forward_on_instance(
                self.classification_reader.text_to_instance(sequence)
            )['probs'][self.label_prob_to_drop]

    @abstractmethod
    def step(self):
        pass

    def sample(self, num_steps: int = 100) -> List[SamplerOutput]:
        for _ in range(num_steps):
            self.step()
        return self.history

    def sample_until_satisfied(self, max_steps: int = 200,
                               bleu_diff: float = -0.2, prob_drop: float = 1.5) -> SamplerOutput:
        """
        Sample until the bleu decreases by `bleu_diff` maximum and the probability drops by `prob_drop` times minimum
        """

        for _ in range(max_steps):
            self.step()
            if self.history and self.history[-1].bleu_diff >= bleu_diff and \
                    self.history[-1].prob_drop >= prob_drop:
                return self.history[-1]
        return SamplerOutput(generated_sequence=self.initial_sequence) if not self.history \
            else max(self.history, key=lambda x: x.prob_drop)


class RandomSampler(Sampler):
    def step(self) -> None:
        assert self.current_state is not None, 'Run `set_input()` first'
        new_state = self.current_state.copy()
        new_state['decoder_hidden'] = self.proposal_distribution.sample(new_state['decoder_hidden'])
        generated_seq = self.generate_from_state(new_state.copy())

        if generated_seq:
            new_prob = self.predict_prob(generated_seq)
            new_bleu = calculate_bleu2(self.initial_sequence, generated_seq)
            prob_diff = new_prob - self.initial_prob
            bleu_diff = new_bleu - self.initial_bleu
            prob_drop = self.initial_prob / (new_prob + 1e-16)
            bleu_drop = self.initial_bleu / (new_bleu + 1e-16)

            self.history.append(
                SamplerOutput(
                    generated_sequence=generated_seq,
                    prob=new_prob,
                    bleu=new_bleu,
                    prob_diff=prob_diff,
                    bleu_diff=bleu_diff,
                    prob_drop=prob_drop,
                    bleu_drop=bleu_drop
                )
            )


class MCMCSampler(Sampler):
    def __init__(
            self,
            proposal_distribution: Proposal,
            classification_model: BasicClassifier,
            classification_reader: DatasetReader,
            generation_model: OneLanguageSeq2SeqModel,
            generation_reader: DatasetReader,
            bleu: bool = True,
            sigma_prob: float = 1,
            sigma_bleu: float = 2,
            device: int = -1
    ) -> None:
        super().__init__(proposal_distribution, classification_model, classification_reader,
                         generation_model, generation_reader, device)
        self.bleu = bleu
        self.sigma_prob = sigma_prob
        self.sigma_bleu = sigma_bleu

    def step(self) -> None:
        assert self.current_state is not None, 'Run `set_input()` first'
        new_state = self.current_state.copy()
        new_state['decoder_hidden'] = self.proposal_distribution.sample(new_state['decoder_hidden'])
        generated_seq = self.generate_from_state(new_state.copy())

        if generated_seq:
            new_prob = self.predict_prob(generated_seq)
            new_bleu = calculate_bleu2(self.initial_sequence, generated_seq)
            prob_diff = new_prob - self.initial_prob
            bleu_diff = new_bleu - self.initial_bleu
            prob_drop = self.initial_prob / (new_prob + 1e-16)
            bleu_drop = self.initial_bleu / (new_bleu + 1e-16)

            exp_base = (-1 / prob_drop) / self.sigma_prob
            if self.bleu:
                exp_base -= bleu_drop / self.sigma_bleu

            acceptance_probability = min(
                [
                    1.,
                    np.exp(exp_base)
                ]
            )

            if acceptance_probability > np.random.rand():
                self.current_state = new_state

                self.history.append(
                    SamplerOutput(
                        generated_sequence=generated_seq,
                        prob=new_prob,
                        bleu=new_bleu,
                        prob_diff=prob_diff,
                        bleu_diff=bleu_diff,
                        prob_drop=prob_drop,
                        bleu_drop=bleu_drop,
                        acceptance_probability=acceptance_probability
                    )
                )
