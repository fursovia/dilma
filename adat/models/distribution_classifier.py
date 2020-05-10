from typing import Dict

import torch
from allennlp.data import TextFieldTensors
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder
from allennlp.nn import util
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.training.metrics import CategoricalAccuracy

from .masked_lm import MaskedLanguageModel


@Model.register("distribution_classifier")
class DistributionClassifier(Model):

    def __init__(
        self,
        masked_lm: MaskedLanguageModel,
        seq2vec_encoder: Seq2VecEncoder,
        num_labels: int,
        dropout: float = None,
        namespace: str = "tokens",
    ) -> None:

        super().__init__(masked_lm.vocab)

        self._masked_lm = masked_lm
        self._masked_lm.eval()
        self._masked_lm._tokens_masker = None

        self._seq2vec_encoder = seq2vec_encoder
        self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self._namespace = namespace

        self._num_labels = num_labels
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

    def forward_on_lm_output(
            self,
            lm_output: Dict[str, torch.Tensor],
            label: torch.IntTensor = None
    ) -> Dict[str, torch.Tensor]:
        embedded_text = self._seq2vec_encoder(torch.softmax(lm_output["logits"], dim=-1), mask=lm_output["mask"])

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        logits = self._classification_layer(embedded_text)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)

        return output_dict

    def forward(
        self, tokens: TextFieldTensors, label: torch.IntTensor = None
    ) -> Dict[str, torch.Tensor]:

        with torch.no_grad():
            lm_output = self._masked_lm(tokens)

        output_dict = self.distribution_to_preds(lm_output, label)
        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(tokens)
        return output_dict

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary(self._label_namespace).get(
                label_idx, str(label_idx)
            )
            classes.append(label_str)
        output_dict["label"] = classes
        tokens = []
        for instance_tokens in output_dict["token_ids"]:
            tokens.append(
                [
                    self.vocab.get_token_from_index(token_id.item(), namespace=self._namespace)
                    for token_id in instance_tokens
                ]
            )
        output_dict["tokens"] = tokens
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self._accuracy.get_metric(reset)}
        return metrics

    @classmethod
    def from_params(cls, params: Params, vocab: Vocabulary, **extras) -> "DistributionClassifier":
        masked_lm_params = params.pop("masked_lm")
        assert masked_lm_params["type"] == "from_archive"
        masked_lm = Model.from_archive(masked_lm_params["archive_file"])

        seq2vec_encoder_params = params.pop("seq2vec_encoder")
        seq2vec_encoder = Seq2VecEncoder.from_params(seq2vec_encoder_params, vocab=vocab)

        num_labels = params.pop_int("num_labels")
        dropout = params.pop_float("dropout", None)

        params.assert_empty(cls.__name__)

        return cls(masked_lm=masked_lm, seq2vec_encoder=seq2vec_encoder, num_labels=num_labels, dropout=dropout)
