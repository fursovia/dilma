from copy import deepcopy

import torch
from allennlp.data import TextFieldTensors

from adat.attackers.attacker import AttackerOutput
from .cascada import Cascada


class DistributionCascada(Cascada):

    def step(
            self,
            inputs: TextFieldTensors,
            sequence_to_attack: str,
            label_to_attack: int,
            initial_prob: float,
            **kwargs
    ) -> AttackerOutput:
        lm_output = self.lm_model(inputs)

        # (self.num_gumbel_samples, )
        prob = self.classifier.forward_on_lm_output(lm_output)["probs"][0, label_to_attack]
        # (self.num_gumbel_samples, )
        distance = self.deep_levenshtein.forward_on_lm_output(
            lm_output, kwargs["initial_lm_output"]
        )["distance"][0, 0]

        loss = self.calculate_loss(
            prob,
            distance
        )
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # (1, sequence_length, vocab_size)
        logits = self.lm_model(inputs)["logits"]
        # max(self.num_samples, 1) adversarial attacks
        adversarial_sequences = self.decode_sequence(logits)

        outputs = []
        for adversarial_sequence in set(adversarial_sequences):
            output = self.get_output(
                sequence_to_attack=sequence_to_attack,
                adversarial_sequence=adversarial_sequence,
                label_to_attack=label_to_attack,
                initial_prob=initial_prob,
                loss_value=loss.item(),
                approx_wer=distance.item(),
                approx_prob=prob.item()
            )
            outputs.append(output)

        return self.find_best_attack(outputs)

    def attack(
            self,
            sequence_to_attack: str,
            label_to_attack: int = 1,
            max_steps: int = 5,
            early_stopping: bool = False
    ) -> AttackerOutput:
        assert max_steps > 0
        inputs = self.sequence_to_input(sequence_to_attack)
        with torch.no_grad():
            prob = self.classifier(inputs)["probs"][0, label_to_attack].item()
            initial_lm_output = self.lm_model(inputs)

        outputs = []
        for _ in range(max_steps):
            output = self.step(
                inputs,
                sequence_to_attack=sequence_to_attack,
                label_to_attack=label_to_attack,
                initial_prob=prob,
                initial_lm_output=initial_lm_output
            )
            outputs.append(output)
            if early_stopping and output.adversarial_label != label_to_attack:
                break

        output = self.find_best_attack(outputs)
        output.history = [deepcopy(o.__dict__) for o in outputs]
        self.initialize_load_state_dict()
        self.initialize_optimizer()
        return output
