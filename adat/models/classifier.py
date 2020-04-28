from typing import Dict, Union

import torch
from allennlp.models import BasicClassifier, Model
from allennlp.nn.util import get_text_field_mask
from allennlp.data import TextFieldTensors

from .deep_levenshtein import OneHot


@Model.register(name="basic_classifier_one_hot_support")
class BasicClassifierOneHotSupport(BasicClassifier):
    def forward(  # type: ignore
        self, tokens: Union[TextFieldTensors, OneHot], label: torch.IntTensor = None
    ) -> Dict[str, torch.Tensor]:

        if isinstance(tokens, OneHot):
            # TODO: sparse tensors support
            embedded_text = torch.matmul(tokens, self._text_field_embedder._token_embedders["tokens"].weight)
            indexes = torch.argmax(tokens, dim=-1)
            mask = (~torch.eq(indexes, 0)).float()
        else:
            embedded_text = self._text_field_embedder(tokens)
            mask = get_text_field_mask(tokens)

        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        logits = self._classification_layer(embedded_text)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)

        return output_dict
