from typing import List, Dict, Optional, Tuple

import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset import Batch
from allennlp.nn.util import move_to_device

from adat.models import MaskedCopyNet, Classifier, DeepLevenshtein
from adat.dataset import CopyNetReader, IDENTITY_TOKEN
from adat.attackers.attacker import AttackerOutput, Attacker, find_best_output
from adat.utils import calculate_wer


BASIC_MASKER = [IDENTITY_TOKEN]


class Cascada(Attacker):
    def __init__(
            self,
            vocab: Vocabulary,
            reader: CopyNetReader,
            classification_model: Classifier,
            masked_copynet: MaskedCopyNet,
            deep_levenshtein_model: DeepLevenshtein,
            levenshtein_weight: float = 0.1,
            learning_rate: float = 0.5,
            num_updates: int = 2,
            num_labels: int = 2,
            device: int = -1
    ) -> None:
        self.vocab = vocab
        self.reader = reader
        self.classification_model = classification_model
        self.masked_copynet = masked_copynet
        self.deep_levenshtein_model = deep_levenshtein_model
        self.device = device
        if self.device >= 0 and torch.cuda.is_available():
            self.classification_model.cuda(self.device)
            self.masked_copynet.cuda(self.device)
            self.deep_levenshtein_model.cuda(self.device)
        else:
            self.classification_model.cpu()
            self.masked_copynet.cpu()
            self.deep_levenshtein_model.cpu()

        self.num_labels = num_labels
        self.levenshtein_weight = levenshtein_weight
        self.learning_rate = learning_rate
        self.num_updates = num_updates

        self.initial_sequence = None
        self.initial_state = None
        self.initial_prob = None
        self.initial_label = None
        self.history: List[AttackerOutput] = list()

    def generate_sequence_from_state(self, state: Dict[str, torch.Tensor]) -> List[str]:
        pred_output = self.masked_copynet.beam_search(state)
        predicted_sequences = []
        for seq in self.masked_copynet.decode(pred_output)['predicted_tokens'][0]:
            predicted_sequences.append(' '.join(seq))
        return predicted_sequences

    def predict_prob_from_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        encdoded_class = self.classification_model._seq2vec_encoder(
            state['encoder_outputs'],
            mask=state['source_mask']
        )
        logits = self.classification_model._classification_layer(encdoded_class)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        output_dict = {"logits": logits, "probs": probs}
        output_dict = self.classification_model.decode(output_dict)
        return output_dict

    def _get_embedded_input(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        vector = self.deep_levenshtein_model.seq2vec_encoder(
            state['encoder_outputs'],
            mask=state['source_mask']
        )
        embedded_input = {
            'mask': state['source_mask'],
            'vector': vector,
            'matrix': state['encoder_outputs']
        }
        return embedded_input

    def calculate_similarity_from_state(
            self,
            state_a: Dict[str, torch.Tensor],
            state_b: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        embedded_input_a = self._get_embedded_input(state_a)
        embedded_input_b = self._get_embedded_input(state_b)
        similarity = self.deep_levenshtein_model.calculate_similarity(
            embedded_input_a,
            embedded_input_b
        )
        return similarity

    def _sequence2batch(
            self,
            seq: str,
            mask_tokens: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, torch.LongTensor]]:
        instance = self.reader.text_to_instance(seq, maskers_applied=mask_tokens)
        batch = Batch([instance])
        batch.index_instances(self.vocab)
        return move_to_device(batch.as_tensor_dict(), self.device)  # tokens, mask_tokens

    def set_label_to_attack(self, label: int) -> None:
        self.label_to_attack = label

    def set_input(self, sequence: str, mask_tokens: Optional[List[str]] = None) -> None:
        self.initial_sequence = sequence
        inputs = self._sequence2batch(sequence, mask_tokens)
        self.initial_state = self.masked_copynet.encode(
            source_tokens=inputs['tokens'],
            mask_tokens=inputs['mask_tokens']
        )

        output = self.predict_prob_from_state(self.initial_state)
        self.initial_prob = output['probs']
        self.initial_label = output['label']

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

        return loss

    @staticmethod
    def _update_hidden(hidden: torch.Tensor, alpha: float, num_updates: int = 1) -> torch.Tensor:
        with torch.no_grad():
            for _ in range(num_updates):
                hidden.data = hidden.data - alpha * hidden.grad
            hidden.grad.zero_()

        return hidden

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

    def step(self):
        state = self.initial_state.copy()
        state_adversarial = {key: tensor.clone() for key, tensor in state.items()}

        state['encoder_outputs'].requires_grad = False
        state_adversarial['encoder_outputs'].requires_grad = True

        # Classifier
        classifier_output = self.predict_prob_from_state(state_adversarial)

        # Deep Levenshtein
        similarity = self.calculate_similarity_from_state(state, state_adversarial)

        # Loss
        loss = self._calculate_loss(
            adversarial_probs=classifier_output['probs'],
            original_probs=self.initial_prob,
            similarity=similarity
        )
        loss.backward()

        state_adversarial['encoder_outputs'] = self._update_hidden(
            state_adversarial['encoder_outputs'],
            self.learning_rate,
            self.num_updates
        )

        generated_sequences = self.generate_sequence_from_state(state_adversarial)

        curr_outputs = list()
        # we generated `beam_size` adversarial examples
        for generated_seq in generated_sequences:
            # sometimes len(generated_seq) = 0
            if generated_seq:
                curr_outputs.append(
                    AttackerOutput(
                        generated_sequence=generated_seq,
                        label=int(classifier_output['label']),
                        wer=calculate_wer(self.initial_sequence, generated_seq)
                    )
                )

        if curr_outputs:
            output = find_best_output(curr_outputs, self.label_to_attack)
            self.history.append(output)







class CascadaAttacker(Attacker):
    def __init__(
            self,
            vocab: Vocabulary,
            reader: CopyNetReader,
            classification_model: Classifier,
            masked_copynet: MaskedCopyNet,
            deep_levenshtein_model: DeepLevenshtein,
            levenshtein_weight: float = 0.1,
            num_labels: int = 2,
            device: int = -1
    ) -> None:
        self.vocab = vocab
        self.reader = reader
        self.classification_model = classification_model
        self.masked_copynet = masked_copynet
        self.deep_levenshtein_model = deep_levenshtein_model
        self.device = device
        if self.device >= 0 and torch.cuda.is_available():
            self.classification_model.cuda(self.device)
            self.masked_copynet.cuda(self.device)
            self.deep_levenshtein_model.cuda(self.device)
        else:
            self.classification_model.cpu()
            self.masked_copynet.cpu()
            self.deep_levenshtein_model.cpu()

        self.num_labels = num_labels
        self.levenshtein_weight = levenshtein_weight



    @staticmethod
    def _update_hidden(hidden: torch.Tensor, alpha: float, num_updates: int = 1) -> torch.Tensor:
        with torch.no_grad():
            for _ in range(num_updates):
                hidden.data = hidden.data - alpha * hidden.grad
            hidden.grad.zero_()

        return hidden

    def _sequences2batch(
            self,
            sequences: List[str],
            maskers: List[List[str]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
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
               mask_tokens: Dict[str, torch.Tensor]) -> List[List[str]]:
        output = self.masked_copynet.beam_search(state)
        predictions = self.masked_copynet.decodlkio9e(output)['predicted_tokens']
        return predictions

    def predict_probs_from_state(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        encdoded_class = self.classification_model._seq2vec_encoder(
            state['encoder_outputs'],
            mask=state['source_mask']
        )
        logits = self.classification_model._classification_layer(encdoded_class)
        return torch.nn.functional.softmax(logits, dim=-1)

    def calculate_similarity(self,
                             state_a: Dict[str, torch.Tensor],
                             state_b: Dict[str, torch.Tensor]) -> torch.Tensor:
        vector_a = self.deep_levenshtein_model._seq2vec_encoder(
            state_a['encoder_outputs'],
            mask=state_a['source_mask']
        )
        vector_b = self.deep_levenshtein_model._seq2vec_encoder(
            state_b['encoder_outputs'],
            mask=state_b['source_mask']
        )
        return 0.5 * (self.deep_levenshtein_model._cosine_sim(vector_a, vector_b) + 1)

    def attack_until_label_is_changed(self, num_steps: int = 200, early_stopping: bool = False):
        pass

    def step(self):
        pass

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
        state = self.masked_copynet.encode(batch['tokens'], mask_tokens=batch['mask_tokens'])
        state_adversarial = {key: tensor.clone() for key, tensor in state.items()}

        state['encoder_outputs'].requires_grad = True
        state_adversarial['encoder_outputs'].requires_grad = True

        # DEEP LEVENSHTEIN
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

        decoded = self.decode(batch['tokens'], state=state_adversarial, mask_tokens=batch['mask_tokens'])
        decoded = [' '.join(d) for d in decoded]
        word_error_rates = [calculate_wer(adv_seq, seq) for adv_seq, seq in zip(decoded, sequences)]
        prob_diffs = [prob - aprob for prob, aprob in zip(probs.detach().cpu().numpy(), new_probs)]

        output = [
            AttackerOutput(
                generated_sequence=aseq,
                label=None,
                prob_diff=prob_diff,
                wer=wer
            )
            for aseq, aprob, prob_diff, wer in zip(decoded, new_probs, prob_diffs, word_error_rates)
        ]
        return output
