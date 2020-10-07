from copy import deepcopy
import random

import torch
from allennlp.models import Model
from allennlp.nn import util

from dilma.attackers.attacker import Attacker, AttackerOutput
from dilma.common import SequenceData, START_TOKEN, END_TOKEN, MASK_TOKEN
from dilma.reader import BasicDatasetReader
from dilma.utils.data import data_to_tensors, decode_indexes


@Attacker.register("fgsm")
class FGSM(Attacker):

    SPECIAL_TOKENS = ("@@UNKNOWN@@", "@@PADDING@@", START_TOKEN, END_TOKEN, MASK_TOKEN)

    def __init__(
        self,
        substitute_classifier: Model,
        target_classifier: Model,
        reader: BasicDatasetReader,
        num_steps: int = 50,
        epsilon: float = 1.0,
        device: int = -1,
    ) -> None:
        super().__init__(
            substitute_classifier=substitute_classifier,
            target_classifier=target_classifier,
            reader=reader,
            device=device
        )
        self.num_steps = num_steps
        self.epsilon = epsilon

        self.emb_layer = util.find_embedding_layer(self.substitute_classifier).weight
        self.special_indexes = [
            self.vocab.get_token_index(token, "tokens") for token in self.SPECIAL_TOKENS
        ]

    def attack(self, data_to_attack: SequenceData) -> AttackerOutput:
        # get inputs to the model
        inputs = data_to_tensors(data_to_attack, reader=self.reader, vocab=self.vocab, device=self.device)

        # get original indexes of a sequence
        orig_indexes = inputs["sequence"]["tokens"]["tokens"]

        # original probability of the true label
        orig_prob = self.get_substitute_clf_probs(inputs)[self.label_to_index(data_to_attack.label)].item()

        # get mask and embeddings
        emb_out = self.substitute_classifier.get_embeddings(sequence=inputs["sequence"])

        # disable gradients using a trick
        embeddings = emb_out["embeddings"].detach()
        embeddings_splitted = [e for e in embeddings[0]]

        outputs = []
        for step in range(self.num_steps):
            # choose random index of embeddings (except for start/end tokens)
            random_idx = random.randint(1, max(1, len(data_to_attack) - 2))
            # only one embedding can be modified
            embeddings_splitted[random_idx].requires_grad = True

            # calculate the loss for current embeddings
            loss = self.substitute_classifier.forward_on_embeddings(
                embeddings=torch.stack(embeddings_splitted, dim=0).unsqueeze(0),
                mask=emb_out["mask"],
                label=inputs["label"],
            )["loss"]
            loss.backward()

            # update the chosen embedding
            embeddings_splitted[random_idx] = (
                embeddings_splitted[random_idx] + self.epsilon * embeddings_splitted[random_idx].grad.data.sign()
            )
            self.substitute_classifier.zero_grad()

            # find the closest embedding for the modified one
            distances = torch.nn.functional.pairwise_distance(embeddings_splitted[random_idx], self.emb_layer)
            # we dont choose special tokens
            distances[self.special_indexes] = 10 ** 16

            # swap embeddings
            closest_idx = distances.argmin().item()
            embeddings_splitted[random_idx] = self.emb_layer[closest_idx]
            embeddings_splitted = [e.detach() for e in embeddings_splitted]

            # get adversarial indexes
            adversarial_idexes = orig_indexes.clone()
            adversarial_idexes[0, random_idx] = closest_idx

            adv_data = deepcopy(data_to_attack)
            adv_data.sequence = decode_indexes(adversarial_idexes[0], vocab=self.vocab)

            adv_inputs = data_to_tensors(adv_data, self.reader, self.vocab, self.device)

            # get adversarial probability and adversarial label
            adv_probs = self.get_substitute_clf_probs(adv_inputs)
            adv_data.label = self.probs_to_label(adv_probs)
            adv_prob = adv_probs[self.label_to_index(data_to_attack.label)].item()

            output = AttackerOutput.from_data(
                data_to_attack=data_to_attack,
                adversarial_data=adv_data,
                probability=orig_prob,
                adversarial_probability=adv_prob
            )
            outputs.append(output)

        best_output = self.find_best_attack(outputs)

        return best_output
