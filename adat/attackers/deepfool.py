"""Deepfool: a simple and accurate method to fool deep neural networks"""

from pathlib import Path
from typing import Optional
from copy import deepcopy
from functools import lru_cache
import random

import torch
from allennlp.models import load_archive
from allennlp.data import TextFieldTensors, Batch, DatasetReader
from allennlp.nn.util import move_to_device
from allennlp.nn import util

from adat.attackers import Attacker, AttackerOutput
from adat.utils import calculate_wer


class DeepFoolAttacker(Attacker):

    def __init__(
            self,
            classifier_dir: str,
            num_steps: int = 10,
            max_steps: int = 10,
            epsilon: float = 1.02,
            device: int = -1
    ) -> None:

        archive = load_archive(Path(classifier_dir) / "model.tar.gz")
        self.reader = DatasetReader.from_params(archive.config["dataset_reader"])
        self.classifier = archive.model
        self.classifier.eval()

        self.num_steps = num_steps
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.device = device

        if self.device >= 0 and torch.cuda.is_available():
            self.classifier.cuda(self.device)

        self.emb_layer = self._construct_embedding_matrix()
        self.num_labels = self.classifier._num_labels
        self.vocab_size = self.classifier.vocab.get_vocab_size()

    def _construct_embedding_matrix(self):
        embedding_layer = util.find_embedding_layer(self.classifier)
        self.embedding_layer = embedding_layer
        return embedding_layer.weight

    def indexes_to_string(self, indexes: torch.Tensor) -> str:
        out = [self.classifier.vocab.get_token_from_index(idx.item()) for idx in indexes]
        out = [o for o in out if o not in ["<START>", "<END>"]]
        return " ".join(out)

    @lru_cache(maxsize=1000)
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
            max_steps: Optional[int] = None,
            num_steps: Optional[int] = None,
            epsilon: Optional[float] = None
    ) -> AttackerOutput:
        seq_length = len(sequence_to_attack.split())
        max_steps = max_steps or self.max_steps
        num_steps = num_steps or self.num_steps
        epsilon = epsilon or self.epsilon
        inputs = self.sequence_to_input(sequence_to_attack)

        # trick to make the variable a leaf variable
        emb_inp = self.classifier.get_embeddings(inputs)
        embs = emb_inp['embedded_text'].detach()
        # probability of the original sequence
        initial_prob = self.classifier.forward_on_embeddings(
            embs
        )["probs"][0, label_to_attack].item()
        embs = [e for e in embs[0]]

        history = []
        # we replace random tokens `num_steps` times
        for i in range(num_steps):
            random_idx = random.randint(1, max(1, seq_length - 2))
            # this embedding will be changed
            cloned_emb = embs[random_idx].clone()
            embs[random_idx].requires_grad = True

            perturbations = []
            adv_pred = label_to_attack
            # let's find final perturbation \hat{r}
            while adv_pred == label_to_attack and len(perturbations) <= max_steps:
                weights = dict()
                delta_probs = dict()

                probs = self.classifier.forward_on_embeddings(
                    torch.stack(embs, dim=0).unsqueeze(0)
                )["probs"][0]

                self.classifier.zero_grad()
                probs[label_to_attack].backward(retain_graph=True)
                # \nabla f_{\hat{k}}, where \hat{k} is `label_to_attack`
                f_k_star_grad = embs[random_idx].grad

                for k in range(self.num_labels):
                    if k != label_to_attack:
                        self.classifier.zero_grad()
                        embs[random_idx].grad = None
                        probs[k].backward(retain_graph=True)

                        # w' = \nabla f_k - \nabla f_{\hat{k}}
                        weights[k] = embs[random_idx].grad - f_k_star_grad
                        # f' = f_k - f_{\hat{k}}
                        delta_probs[k] = probs[label_to_attack] - probs[k].item()

                # |f'| / || w' ||_2^2 for all k
                coefs = {
                    k: abs(delta_probs[k]) / torch.norm(weights[k], p=2.0) ** 2
                    for k in weights.keys()
                }

                # k with the minimum |f'| / || w' ||_2^2
                l_star = min(coefs)
                perturbation = coefs[l_star] * weights[l_star]
                perturbations.append(perturbation)

                embs[random_idx] = embs[random_idx] + perturbation
                embs = [e.detach() for e in embs]
                embs[random_idx].requires_grad = True

                adv_out = self.classifier.forward_on_embeddings(
                    torch.stack(embs, dim=0).unsqueeze(0)
                )
                adv_pred = adv_out["probs"][0].argmax().item()

            final_perturbation = torch.stack(perturbations, dim=0).sum(dim=0)
            embs[random_idx] = cloned_emb + epsilon * final_perturbation

            distances = torch.nn.functional.pairwise_distance(
                embs[random_idx],
                self.emb_layer
            )
            # @UNK@, @PAD@, @MASK@, @START@, @END@
            to_drop_indexes = [0, 1] + list(range(self.vocab_size - 3, self.vocab_size))
            distances[to_drop_indexes] = 10e6
            closest_idx = distances.argmin().item()

            embs[random_idx] = self.emb_layer[closest_idx]
            embs = [e.detach() for e in embs]

            adversarial_idexes = inputs["tokens"]["tokens"].clone()
            adversarial_idexes[0, random_idx] = closest_idx

            adverarial_seq = self.indexes_to_string(adversarial_idexes[0])
            new_clf_output = self.classifier.forward(self.sequence_to_input(adverarial_seq))
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
