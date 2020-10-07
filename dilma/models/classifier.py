from typing import Dict, Optional

import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure
from allennlp.nn.util import get_text_field_mask
from allennlp.nn import util


@Model.register("sequence_classifier")
class SequenceClassifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        sequence_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        seq2seq_encoder: Optional[Seq2SeqEncoder] = None,
        num_labels: Optional[int] = None,
    ) -> None:

        super().__init__(vocab)
        self._sequence_field_embedder = sequence_field_embedder
        self._seq2vec_encoder = seq2vec_encoder
        self._seq2seq_encoder = seq2seq_encoder

        num_labels = num_labels or vocab.get_vocab_size("labels")
        self._classification_layer = torch.nn.Linear(self._seq2vec_encoder.get_output_dim(), num_labels)

        self._loss = torch.nn.CrossEntropyLoss()
        self._accuracy = CategoricalAccuracy()
        self._f1 = FBetaMeasure(beta=1.0)

    def get_embeddings(self, sequence: TextFieldTensors) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sequence)
        embeddings = self._sequence_field_embedder(sequence)
        return {"mask": mask, "embeddings": embeddings}

    def forward_on_embeddings(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
        label: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        if self._seq2seq_encoder is not None:
            embeddings = self._seq2seq_encoder(embeddings, mask=mask)

        contextual_embeddings = self._seq2vec_encoder(embeddings, mask=mask)

        logits = self._classification_layer(contextual_embeddings)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = dict(logits=logits, probs=probs)
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)
            self._f1(logits, label)

        return output_dict

    def forward(
        self,
        sequence: TextFieldTensors,
        label: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        emb_out = self.get_embeddings(sequence)
        output_dict = self.forward_on_embeddings(
            embeddings=emb_out["embeddings"],
            mask=emb_out["mask"],
            label=label,
        )
        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(sequence)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "accuracy": self._accuracy.get_metric(reset),
            "f1": self._f1.get_metric(reset)
        }
        return metrics
