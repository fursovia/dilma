from typing import Dict, Optional, Union

import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.data import TextFieldTensors
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn import util


OneHot = torch.Tensor


def get_onehot_mask(tensors: OneHot) -> torch.BoolTensor:
    indexes = torch.argmax(tensors, dim=-1)
    return torch.eq(indexes, 0).type(torch.bool)


@Model.register(name="deep_levenshtein")
class DeepLevenshtein(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        seq2seq_encoder: Optional[Seq2SeqEncoder] = None,
    ) -> None:
        super().__init__(vocab)
        self.text_field_embedder = text_field_embedder
        self.seq2seq_encoder = seq2seq_encoder
        self.seq2vec_encoder = seq2vec_encoder
        self.linear = torch.nn.Linear(self.seq2vec_encoder.get_output_dim() * 3, 1)
        self._loss = torch.nn.MSELoss()

    def encode_sequence(self, sequence: Union[OneHot, TextFieldTensors]) -> torch.Tensor:

        if isinstance(sequence, OneHot):
            # TODO: sparse tensors support
            embedded_sequence = torch.matmul(sequence, self.text_field_embedder._token_embedders["tokens"].weight)
            indexes = torch.argmax(sequence, dim=-1)
            mask = torch.eq(indexes, 0).type(torch.bool)
        else:
            embedded_sequence = self.text_field_embedder(sequence)
            mask = util.get_text_field_mask(sequence).float()
        # It is needed if we pad the initial sequence (or truncate)
        mask = torch.nn.functional.pad(mask, pad=[0, embedded_sequence.size(1) - mask.size(1)])
        if self.seq2seq_encoder is not None:
            embedded_sequence = self.seq2seq_encoder(embedded_sequence, mask=mask)
        embedded_sequence_vector = self.seq2vec_encoder(embedded_sequence, mask=mask)
        return embedded_sequence_vector

    def forward(
        self,
        sequence_a: Union[OneHot, TextFieldTensors],
        sequence_b: Union[OneHot, TextFieldTensors],
        distance: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        embedded_sequence_a = self.encode_sequence(sequence_a)
        embedded_sequence_b = self.encode_sequence(sequence_b)
        diff = torch.abs(embedded_sequence_a - embedded_sequence_b)

        representation = torch.cat([embedded_sequence_a, embedded_sequence_b, diff], dim=-1)
        approx_distance = self.linear(representation)
        output_dict = {"distance": approx_distance}

        if distance is not None:
            output_dict["loss"] = self._loss(approx_distance.view(-1), distance.view(-1))
        return output_dict
