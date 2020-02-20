from typing import List, Dict, Optional, Tuple
from functools import lru_cache

import torch
from allennlp.data.vocabulary import Vocabulary

from adat.models import MaskedCopyNet, Classifier, DeepLevenshtein
from adat.dataset import CopyNetReader, IDENTITY_TOKEN, ClassificationReader
from adat.attackers.attacker import Attacker, AttackerOutput, find_best_output
from adat.utils import calculate_wer, calculate_normalized_wer


BASIC_MASKER = [IDENTITY_TOKEN]


class Cascada(Attacker):
    def __init__(
            self,
            vocab: Vocabulary,
            reader: CopyNetReader,
            class_reader: ClassificationReader,
            classification_model: Classifier,
            classification_model_basic: Classifier,
            masked_copynet: MaskedCopyNet,
            deep_levenshtein_model: DeepLevenshtein,
            levenshtein_weight: float = 0.1,
            prob_diff_weight: float = 1.0,
            learning_rate: float = 0.5,
            num_updates: int = 2,
            num_labels: int = 2,
            device: int = -1
    ) -> None:
        super().__init__(device=device)
        self.vocab = vocab
        self.reader = reader
        self.classification_reader = class_reader
        self.classification_model = classification_model
        self.classification_model_basic = classification_model_basic
        self.masked_copynet = masked_copynet
        self.deep_levenshtein_model = deep_levenshtein_model
        if self.device >= 0 and torch.cuda.is_available():
            self.classification_model.cuda(self.device)
            self.masked_copynet.cuda(self.device)
            self.deep_levenshtein_model.cuda(self.device)
            self.classification_model_basic.cuda(self.device)
        else:
            self.classification_model.cpu()
            self.masked_copynet.cpu()
            self.deep_levenshtein_model.cpu()
            self.classification_model_basic.cpu()

        self.num_labels = num_labels
        self.levenshtein_weight = levenshtein_weight
        self.prob_diff_weight = prob_diff_weight
        self.learning_rate = learning_rate
        self.num_updates = num_updates

        self.initial_state = None
        self.initial_prob_from_seq = None

    @lru_cache(maxsize=1000)
    def predict_prob_and_label(self, sequence: str) -> Tuple[float, int]:
        assert self.label_to_attack is not None, 'You must run `.set_label_to_attack()` first.'
        # logits, probs, label
        with torch.no_grad():
            predictions = self.classification_model_basic.forward_on_instance(
                self.classification_reader.text_to_instance(sequence)
            )
            prob = predictions['probs'][self.label_to_attack]
            label = predictions['label']
            return float(prob), int(label)

    @lru_cache(maxsize=1000)
    def get_output(self, generated_sequence: str) -> AttackerOutput:
        new_prob, new_label = self.predict_prob_and_label(generated_sequence)
        new_wer = calculate_wer(self.initial_sequence, generated_sequence)
        prob_diff = self.initial_prob_from_seq - new_prob
        return AttackerOutput(
            sequence=self.initial_sequence,
            label=self.label_to_attack,
            adversarial_sequence=generated_sequence,
            adversarial_label=new_label,
            wer=new_wer,
            prob_diff=prob_diff
        )

    def generate_sequence_from_state(self, state: Dict[str, torch.Tensor]) -> List[str]:
        state = self.masked_copynet.init_decoder_state(state)
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

    def set_input(self, sequence: str, mask_tokens: Optional[List[str]] = None) -> None:
        self.initial_sequence = sequence
        inputs = self._sequence2batch(
            sequence=sequence,
            reader=self.reader,
            vocab=self.vocab,
            mask_tokens=mask_tokens
        )
        with torch.no_grad():
            self.initial_state = self.masked_copynet.encode(
                source_tokens=inputs['tokens'],
                mask_tokens=inputs['mask_tokens']
            )
            self.current_state = self.initial_state.copy()
            self.initial_prob = self.predict_prob_from_state(self.initial_state)['probs']
            generated_sequences = self.generate_sequence_from_state(self.initial_state.copy())
            self.initial_prob_from_seq = self.predict_prob_and_label(generated_sequences[0])[0]

    def empty_history(self) -> None:
        super().empty_history()
        self.initial_state = None
        self.initial_prob_from_seq = None

    def _calculate_loss(self,
                        adversarial_probs: torch.Tensor,
                        original_probs: torch.Tensor,
                        similarity: torch.Tensor) -> torch.Tensor:
        loss = torch.add(
            self.prob_diff_weight * torch.sub(
                1,
                torch.sub(
                    original_probs[0][self.label_to_attack],
                    adversarial_probs[0][self.label_to_attack]
                )
            ),
            self.levenshtein_weight * torch.sub(1, similarity[0])
        )

        return loss

    @staticmethod
    def _update_hidden(hidden: torch.Tensor, alpha: float, num_updates: int = 1) -> torch.Tensor:
        with torch.no_grad():
            for _ in range(num_updates):
                hidden.data = hidden.data - alpha * hidden.grad
            hidden.grad.zero_()

        return hidden

    def step(self):
        state_adversarial = {key: tensor.clone() for key, tensor in self.current_state.items()}
        state_adversarial['encoder_outputs'].requires_grad = True

        # Classifier [GRAD]
        classifier_output = self.predict_prob_from_state(state_adversarial)

        # Deep Levenshtein [GRAD]
        similarity = self.calculate_similarity_from_state(self.initial_state, state_adversarial)

        # Loss [GRAD]
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

        # We need to calculate probability one again
        with torch.no_grad():
            # classifier_output = self.predict_prob_from_state(state_adversarial)
            generated_sequences = self.generate_sequence_from_state(state_adversarial.copy())

        curr_outputs = list()
        # we generated `beam_size` adversarial examples
        for generated_seq in generated_sequences:

            # sometimes len(generated_seq) = 0
            if generated_seq:
                curr_outputs.append(self.get_output(generated_seq))
                # curr_outputs.append(
                #     AttackerOutput(
                #         sequence=self.initial_sequence,
                #         label=self.label_to_attack,
                #         adversarial_sequence=generated_seq,
                #         adversarial_label=int(classifier_output['label'][0]),
                #         wer=calculate_wer(self.initial_sequence, generated_seq),
                #         prob_diff=(
                #                 self.initial_prob[0][self.label_to_attack] -
                #                 classifier_output['probs'][0][self.label_to_attack]
                #         ).item()
                #     )
                # )

        if curr_outputs:
            for _, val in state_adversarial.items():
                val.requires_grad = False
            self.current_state = state_adversarial.copy()

            output = find_best_output(curr_outputs, self.label_to_attack)
            self.history.append(output)
