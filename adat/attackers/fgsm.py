"""Explaining and Harnessing Adversarial Examples.
Generating Natural Language Adversarial Examples on a Large Scale with Generative Models"""

from pathlib import Path
from typing import Optional

import torch
from allennlp.models import load_archive
from allennlp.data import TextFieldTensors, Batch, DatasetReader
from allennlp.nn.util import move_to_device
from allennlp.nn import util

from adat.attackers import Attacker, AttackerOutput
from adat.utils import calculate_wer


class FGSMAttacker(Attacker):

    def __init__(self, classifier_dir: str, epsilon: float = 1e-2, device: int = -1):

        archive = load_archive(Path(classifier_dir) / "model.tar.gz")
        self.reader = DatasetReader.from_params(archive.config["dataset_reader"])
        self.classifier = archive.model
        self.classifier.eval()

        self.epsilon = epsilon
        self.device = device

        if self.device >= 0 and torch.cuda.is_available():
            self.classifier.cuda(self.device)

        self.emb_layer = self._construct_embedding_matrix()

    def _construct_embedding_matrix(self):
        embedding_layer = util.find_embedding_layer(self.classifier)
        self.embedding_layer = embedding_layer
        return embedding_layer.weight

    def indexes_to_string(self, indexes: torch.Tensor) -> str:
        out = [self.classifier.vocab.get_token_from_index(idx.item()) for idx in indexes]
        out = [o for o in out if o not in ["<START>", "<END>"]]
        return " ".join(out)

    def decode_from_closest(self, embeddings: torch.Tensor) -> str:
        embeddings = embeddings[0]
        closest_idx = torch.stack(
            [
                torch.nn.functional.pairwise_distance(
                    embeddings[i],
                    self.emb_layer
                ).argmin()
                for i in range(embeddings.size(0))
            ]
        )

        return self.indexes_to_string(closest_idx)

    def sequence_to_input(self, sequence: str) -> TextFieldTensors:
        instances = Batch([
            self.reader.text_to_instance(sequence)
        ])

        instances.index_instances(self.lm_model.vocab)
        inputs = instances.as_tensor_dict()["tokens"]
        return move_to_device(inputs, self.device)

    def attack(self, sequence_to_attack: str,
               label_to_attack: int = 1, epsilon: Optional[float] = None) -> AttackerOutput:
        epsilon = epsilon or self.epsilon
        inputs = self.sequence_to_input(sequence_to_attack)

        # trick to make the variable a leaf variable
        embs = inputs['embedded_text'].detach()
        embs.requires_grad = True

        clf_output = self.classifier.forward_on_embeddings(
            embs,
            inputs["mask"],
            label=torch.tensor([label_to_attack], device=embs.device)
        )

        loss = clf_output["loss"]
        self.classifier.zero_grad()
        loss.backward()

        perturbed_embs = embs + epsilon * embs.grad.data.sign()
        adverarial_seq = self.decode_from_closest(perturbed_embs)
        new_clf_output = self.classifier.forward(self.sequence_to_input(adverarial_seq))

        initial_prob = clf_output["probs"][0, label_to_attack].item()
        new_probs = new_clf_output["probs"]
        adv_prob = new_probs[0, label_to_attack].item()
        output = AttackerOutput(
            sequence=sequence_to_attack,
            probability=initial_prob,
            adversarial_sequence=adverarial_seq,
            adversarial_probability=adv_prob,
            wer=calculate_wer(sequence_to_attack, adverarial_seq),
            prob_diff=(initial_prob - adv_prob),
            attacked_label=label_to_attack,
            adversarial_label=new_probs.argmax().item()
        )

        return output
