from typing import Dict, Optional

import torch
from allennlp.models.model import Model
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.data import TextFieldTensors
from allennlp.common import Params
from allennlp.data import Vocabulary

from .masked_lm import MaskedLanguageModel


@Model.register(name="distribution_deep_levenshtein")
class DistributionDeepLevenshtein(Model):
    def __init__(
        self,
        masked_lm: MaskedLanguageModel,
        seq2vec_encoder: Seq2VecEncoder,
    ) -> None:
        super().__init__(masked_lm.vocab)
        self._masked_lm = masked_lm
        self._masked_lm.eval()
        self._masked_lm._tokens_masker = None

        self.seq2vec_encoder = seq2vec_encoder
        self.linear = torch.nn.Linear(self.seq2vec_encoder.get_output_dim() * 3, 1)
        self._loss = torch.nn.MSELoss()

    def encode_sequence(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        distribution = torch.softmax(logits, dim=-1)
        embedded_sequence_vector = self.seq2vec_encoder(distribution, mask=mask)
        return embedded_sequence_vector

    def forward_on_lm_output(
            self,
            lm_output_a: Dict[str, torch.Tensor],
            lm_output_b: Dict[str, torch.Tensor],
            distance: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        embedded_sequence_a = self.encode_sequence(lm_output_a["logits"], lm_output_a["mask"])
        embedded_sequence_b = self.encode_sequence(lm_output_b["logits"], lm_output_b["mask"])
        diff = torch.abs(embedded_sequence_a - embedded_sequence_b)

        representation = torch.cat([embedded_sequence_a, embedded_sequence_b, diff], dim=-1)
        approx_distance = self.linear(representation)
        output_dict = {"distance": approx_distance}

        if distance is not None:
            output_dict["loss"] = self._loss(approx_distance.view(-1), distance.view(-1))
        return output_dict

    def forward(
        self,
        sequence_a: TextFieldTensors,
        sequence_b: TextFieldTensors,
        distance: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        with torch.no_grad():
            lm_output_a = self._masked_lm(sequence_a)
            lm_output_b = self._masked_lm(sequence_b)

        output_dict = self.forward_on_lm_output(lm_output_a, lm_output_b)
        return output_dict

    @classmethod
    def from_params(cls, params: Params, vocab: Vocabulary, **extras) -> "DistributionDeepLevenshtein":
        masked_lm_params = params.pop("masked_lm")
        assert masked_lm_params["type"] == "from_archive"
        masked_lm = Model.from_archive(masked_lm_params["archive_file"])

        seq2vec_encoder_params = params.pop("seq2vec_encoder")
        seq2vec_encoder = Seq2VecEncoder.from_params(seq2vec_encoder_params, vocab=vocab)

        params.assert_empty(cls.__name__)

        return cls(masked_lm=masked_lm, seq2vec_encoder=seq2vec_encoder)
