from typing import List, Dict, Optional

import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset import Batch
from allennlp.nn.util import move_to_device

from adat.models.classification_model import BasicClassifierWithMetric
from adat.models.seq2seq_model import OneLanguageSeq2SeqModel
from adat.models.deep_levenshtein import DeepLevenshteinFromSeq2Seq
from adat.dataset import OneLangSeq2SeqReader, IDENTITY_TOKEN
from adat.attackers.attacker import AttackerOutput, Attacker
from adat.utils import calculate_wer


BASIC_MASKER = [IDENTITY_TOKEN]


class GradientAttacker(Attacker):
    def __init__(self,
                 vocab: Vocabulary,
                 reader: OneLangSeq2SeqReader,
                 classification_model: BasicClassifierWithMetric,
                 seq2seq_model: OneLanguageSeq2SeqModel,
                 deep_levenshtein_model: DeepLevenshteinFromSeq2Seq,
                 levenshtein_weight: float = 0.1,
                 num_labels: int = 2,
                 device: int = -1) -> None:
        self.vocab = vocab
        self.reader = reader
        self.classification_model = classification_model
        self.seq2seq_model = seq2seq_model
        self.deep_levenshtein_model = deep_levenshtein_model
        self.device = device
        if self.device >= 0 and torch.cuda.is_available():
            self.classification_model.cuda(self.device)
            self.seq2seq_model.cuda(self.device)
            self.deep_levenshtein_model.cuda(self.device)
        else:
            self.classification_model.cpu()
            self.seq2seq_model.cpu()
            self.deep_levenshtein_model.cpu()

        self.num_labels = num_labels
        self.levenshtein_weight = levenshtein_weight

    @staticmethod
    def _update_hidden(hidden: torch.Tensor, alpha: float, num_updates: int) -> torch.Tensor:
        with torch.no_grad():
            for _ in range(num_updates):
                hidden.data = hidden.data - alpha * hidden.grad
            hidden.grad.zero_()

        return hidden

    def _sequences2batch(self, sequences: List[str], maskers: List[List[str]]) -> Dict[str, Dict[str, torch.Tensor]]:
        instances = [self.reader.text_to_instance(seq, masker) for seq, masker in zip(sequences, maskers)]
        batch = Batch(instances)
        batch.index_instances(self.vocab)
        return move_to_device(batch.as_tensor_dict(), self.device)

    def _get_slicing_index(self, labels_to_attack: List[int], device) -> torch.Tensor:
        # device = None if self.device < 0 else self.device
        indexes = torch.nn.functional.one_hot(
            torch.tensor(labels_to_attack, dtype=torch.int64, device=device),
            num_classes=self.num_labels
        ).type(torch.bool)

        return indexes

    def _calculate_loss(self,
                        adversarial_probs: torch.Tensor,
                        original_probs: torch.Tensor,
                        similarity: torch.Tensor) -> torch.Tensor:
        loss = torch.add(
            torch.sub(
                1,
                torch.sub(original_probs, adversarial_probs)
            ),
            self.levenshtein_weight * torch.sub(1, similarity)
        )

        return loss.mean()

    def decode(self, source_tokens: Dict[str, torch.Tensor],
               state: Dict[str, torch.Tensor],
               masker_tokens: Dict[str, torch.Tensor]) -> List[List[str]]:
        output = self.seq2seq_model.forward(source_tokens, state=state, masker_tokens=masker_tokens)
        return self.seq2seq_model.decode(output)['predicted_tokens']

    def predict_probs_from_state(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        encdoded_class = self.classification_model._seq2vec_encoder(
            state['encoder_outputs'],
            mask=state['source_mask']
        )
        logits = self.classification_model._classification_layer(encdoded_class)
        return torch.nn.functional.softmax(logits, dim=-1)

    def calculate_similarity(self,
                             tate_a: Dict[str, torch.Tensor],
                             state_b: Dict[str, torch.Tensor]) -> torch.Tensor:
        vector_a = self.deep_levenshtein_model._seq2vec_encoder(
            tate_a['encoder_outputs'],
            mask=tate_a['source_mask']
        )
        vector_b = self.deep_levenshtein_model._seq2vec_encoder(
            state_b['encoder_outputs'],
            mask=state_b['source_mask']
        )
        return 0.5 * (self.deep_levenshtein_model._cosine_sim(vector_a, vector_b) + 1)

    def attack(self,
               sequences: List[str],
               labels: List[int],
               maskers: Optional[List[List[str]]] = None,
               learning_rate: float = 0.1,
               num_updates: int = 5) -> List[AttackerOutput]:
        maskers = maskers or [BASIC_MASKER] * len(sequences)
        batch = self._sequences2batch(sequences, maskers)

        # SEQ2SEQ
        # keys: source_mask, encoder_outputs
        state = self.seq2seq_model._encode(batch['tokens'])
        state_adversarial = {key: tensor.clone() for key, tensor in state.items()}

        state['encoder_outputs'].requires_grad = True
        state_adversarial['encoder_outputs'].requires_grad = True

        # DEEP LEVENSHTEIN
        # TODO: we can skip this part
        similarity = self.calculate_similarity(state, state_adversarial)

        # CLASSIFICATION
        indexes = self._get_slicing_index(labels, similarity.device)
        probs = self.predict_probs_from_state(state).masked_select(indexes)
        probs_adversarial = self.predict_probs_from_state(state_adversarial).masked_select(indexes)

        loss = self._calculate_loss(probs_adversarial, probs, similarity)
        loss.backward()

        state_adversarial['encoder_outputs'] = self._update_hidden(
            state_adversarial['encoder_outputs'],
            learning_rate,
            num_updates
        )

        with torch.no_grad():
            new_probs = self.predict_probs_from_state(state_adversarial).masked_select(indexes).cpu().numpy()

        decoded = self.decode(batch['tokens'], state=state_adversarial, masker_tokens=batch['masker_tokens'])
        decoded = [' '.join(d) for d in decoded]
        # word_error_rates = [calculate_wer(adv_seq, seq) for adv_seq, seq in zip(decoded, sequences)]
        # just to speed up, we can calculate wer later
        word_error_rates = [0.0] * len(decoded)
        prob_diffs = [prob - aprob for prob, aprob in zip(probs.detach().cpu().numpy(), new_probs)]

        output = [
            AttackerOutput(
                generated_sequence=aseq,
                prob=aprob,
                prob_diff=prob_diff,
                wer=wer
            )
            for aseq, aprob, prob_diff, wer in zip(decoded, new_probs, prob_diffs, word_error_rates)
        ]
        return output
