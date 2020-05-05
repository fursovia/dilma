from pathlib import Path
from typing import Tuple, Optional, List
from copy import deepcopy

import torch
from torch.optim import SGD
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.params import Params
from allennlp.data.batch import Batch
from allennlp.data import TextFieldTensors
from allennlp.nn.util import move_to_device

from adat.attackers.attacker import Attacker, AttackerOutput
from adat.models.deep_levenshtein import DeepLevenshtein
from adat.models.classifier import BasicClassifierOneHotSupport
from adat.utils import calculate_wer

_MAX_NUM_LAYERS = 30
PARAMETERS = {
    f"layer_{i}": f"_seq2seq_encoder._transformer.layers.{i}"
    for i in range(_MAX_NUM_LAYERS)
}
PARAMETERS.update({"linear": "_head.linear", "all": ""})


class MaskedCascada(Attacker):

    def __init__(
            self,
            masked_lm_dir: str,
            classifier_dir: str,
            deep_levenshtein_dir: str,
            alpha: float = 2.0,
            lr: float = 0.1,
            num_gumbel_samples: int = 3,
            parameters_to_update: Optional[Tuple[str, ...]] = None,
            device: int = -1
    ) -> None:
        assert num_gumbel_samples >= 1
        masked_lm_dir = Path(masked_lm_dir)
        classifier_dir = Path(classifier_dir)
        deep_levenshtein_dir = Path(deep_levenshtein_dir)

        lm_params = Params.from_file(masked_lm_dir / "config.json")
        self.reader = DatasetReader.from_params(lm_params["dataset_reader"])

        self.lm_model = Model.from_params(
            params=lm_params["model"],
            vocab=Vocabulary.from_files(masked_lm_dir / "vocabulary")
        )
        # TODO: should be fixed
        self.lm_model._tokens_masker = None
        # initial LM weights
        self._lm_state = torch.load(masked_lm_dir / "best.th", map_location="cpu")
        self.initialize_load_state_dict()

        self.classifier = BasicClassifierOneHotSupport.from_archive(classifier_dir / "model.tar.gz")
        self.deep_levenshtein = DeepLevenshtein.from_archive(deep_levenshtein_dir / "model.tar.gz")

        self.lm_model.eval()
        self.classifier.eval()
        self.deep_levenshtein.eval()

        self.device = device
        if self.device >= 0 and torch.cuda.is_available():
            self.lm_model.cuda(self.device)
            self.classifier.cuda(self.device)
            self.deep_levenshtein.cuda(self.device)

        self.alpha = alpha
        self.lr = lr
        self.num_gumbel_samples = num_gumbel_samples
        self.parameters_to_update = parameters_to_update or ("all", )
        self.optimizer = None
        self.initialize_optimizer()

    def initialize_load_state_dict(self) -> None:
        self.lm_model.load_state_dict(self._lm_state)
        self.lm_model.eval()

    def initialize_optimizer(self) -> None:
        parameters = []
        for name in self.parameters_to_update:
            for layer_name, params in self.lm_model.named_parameters():
                if layer_name.startswith(PARAMETERS[name]):
                    parameters.append(params)
        self.optimizer = SGD(parameters, self.lr)

    def sequence_to_input(self, sequence: str) -> TextFieldTensors:
        instances = Batch([
            self.reader.text_to_instance(sequence)
        ])

        instances.index_instances(self.lm_model.vocab)
        inputs = instances.as_tensor_dict()["tokens"]
        return move_to_device(inputs, self.device)

    def calculate_loss(self, prob: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        return torch.pow(torch.sub(1.0, distance), 2.0) + self.alpha * prob

    def indexes_to_string(self, indexes: torch.Tensor):
        out = [self.lm_model.vocab.get_token_from_index(idx.item()) for idx in indexes]
        out = [o for o in out if o not in ["<START>", "<END>"]]
        return " ".join(out)

    def decode_sequence(self, logits: torch.Tensor, sample: bool = False, num_samples: int = 5) -> List[str]:
        if sample:
            out = []
            for _ in range(num_samples):
                indexes = torch.nn.functional.gumbel_softmax(logits[0]).argmax(dim=-1)
                out.append(self.indexes_to_string(indexes))
        else:
            # only one sample with argmax
            indexes = logits[0].argmax(dim=-1)
            out = [self.indexes_to_string(indexes)]

        return out

    def get_output(
            self,
            sequence_to_attack: str,
            adversarial_sequence: str,
            label_to_attack: int,
            initial_prob: float
    ) -> AttackerOutput:
        new_probs = self.classifier(self.sequence_to_input(adversarial_sequence))["probs"][0]
        new_prob = new_probs[label_to_attack].item()
        distance = calculate_wer(adversarial_sequence, sequence_to_attack)

        output = AttackerOutput(
            sequence=sequence_to_attack,
            probability=initial_prob,
            adversarial_sequence=adversarial_sequence,
            adversarial_probability=new_prob,
            wer=distance,
            prob_diff=(initial_prob - new_prob),
            attacked_label=label_to_attack,
            adversarial_label=new_probs.argmax().item()
        )
        return output

    def step(
            self,
            inputs: TextFieldTensors,
            sequence_to_attack: str,
            label_to_attack: int,
            initial_prob: float,
    ) -> AttackerOutput:
        logits = self.lm_model(inputs)["logits"]

        probs = []
        distances = []
        for _ in range(self.num_gumbel_samples):
            onehot_with_gradients = torch.nn.functional.gumbel_softmax(logits, hard=True)
            probs.append(self.classifier(onehot_with_gradients)["probs"][0, label_to_attack])
            distances.append(self.deep_levenshtein(onehot_with_gradients, inputs)["distance"][0, 0])

        loss = self.calculate_loss(
            torch.stack(probs).mean(),
            torch.stack(distances).mean()
        )
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        logits = self.lm_model(inputs)["logits"]
        adversarial_sequences = self.decode_sequence(logits)

        outputs = []
        for adversarial_sequence in set(adversarial_sequences):
            output = self.get_output(
                sequence_to_attack=sequence_to_attack,
                adversarial_sequence=adversarial_sequence,
                label_to_attack=label_to_attack,
                initial_prob=initial_prob
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
        prob = self.classifier(inputs)["probs"][0, label_to_attack].item()

        outputs = []
        for _ in range(max_steps):
            output = self.step(
                inputs,
                sequence_to_attack=sequence_to_attack,
                label_to_attack=label_to_attack,
                initial_prob=prob
            )
            outputs.append(output)
            if early_stopping and output.adversarial_label != label_to_attack:
                break

        output = self.find_best_attack(outputs)
        output.history = [deepcopy(o.__dict__) for o in outputs]
        self.initialize_load_state_dict()
        self.initialize_optimizer()
        return output
