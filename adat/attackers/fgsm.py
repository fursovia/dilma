"""Explaining and Harnessing Adversarial Examples.
Generating Natural Language Adversarial Examples on a Large Scale with Generative Models"""

from pathlib import Path
from typing import Optional
from copy import deepcopy
import random

import torch
from allennlp.models import load_archive
from allennlp.data import TextFieldTensors, Batch, DatasetReader
from allennlp.nn.util import move_to_device
from allennlp.nn import util

from adat.attackers import Attacker, AttackerOutput
from adat.utils import calculate_wer


class FGSMAttacker(Attacker):

    def __init__(self, classifier_dir: str, num_steps: int = 10, epsilon: float = 0.01, device: int = -1):

        archive = load_archive(Path(classifier_dir) / "model.tar.gz")
        self.reader = DatasetReader.from_params(archive.config["dataset_reader"])
        self.classifier = archive.model
        self.classifier.eval()

        self.num_steps = num_steps
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

    def sequence_to_input(self, sequence: str) -> TextFieldTensors:
        instances = Batch([
            self.reader.text_to_instance(sequence)
        ])

        instances.index_instances(self.classifier.vocab)
        inputs = instances.as_tensor_dict()["tokens"]
        return move_to_device(inputs, self.device)

    def attack(
            self,
            sequence_to_attack: str,
            label_to_attack: int = 1,
            num_steps: Optional[int] = None,
            epsilon: Optional[float] = None
    ) -> AttackerOutput:
        seq_length = len(sequence_to_attack.split())
        num_steps = num_steps or self.num_steps
        epsilon = epsilon or self.epsilon
        inputs = self.sequence_to_input(sequence_to_attack)

        # trick to make the variable a leaf variable
        emb_inp = self.classifier.get_embeddings(inputs)
        embs = emb_inp['embedded_text'].detach()
        label = torch.tensor([label_to_attack], device=embs.device)
        embs = [e for e in embs[0]]

        history = []
        for i in range(num_steps):
            random_idx = random.randint(1, seq_length - 2)
            embs[random_idx].requires_grad = True
            embeddings_tensor = torch.stack(embs, dim=0).unsqueeze(0)

            clf_output = self.classifier.forward_on_embeddings(
                embeddings_tensor,
                emb_inp["mask"],
                label=label
            )

            loss = clf_output["loss"]
            self.classifier.zero_grad()
            loss.backward()

            embs[random_idx] = embs[random_idx] + epsilon * embs[random_idx].grad.data.sign()
            closest_idx = torch.nn.functional.pairwise_distance(
                embs[random_idx],
                self.emb_layer
            ).argmin().item()
            embs[random_idx] = self.emb_layer[closest_idx]
            for e in embs:
                e.detach()

            adversarial_idexes = inputs["tokens"]["tokens"].clone()
            adversarial_idexes[0, random_idx] = closest_idx

            adverarial_seq = self.indexes_to_string(adversarial_idexes)
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

            history.append(output)

        output = self.find_best_attack(history)
        output.history = [deepcopy(o.__dict__) for o in history]
        return output
