from typing import Dict, Union

import torch
from allennlp.models import BasicClassifier, Model
from allennlp.nn.util import get_text_field_mask, get_token_ids_from_text_field_tensors
from allennlp.data import TextFieldTensors

from .deep_levenshtein import OneHot


@Model.register(name="basic_classifier_one_hot_support")
class BasicClassifierOneHotSupport(BasicClassifier):

    def forward_on_embeddings(self, embedded_text: torch.Tensor,
                              mask: torch.Tensor = None,
                              label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        if mask is None:
            mask = torch.ones_like(embedded_text, dtype=torch.bool, device=embedded_text.device)

        output_dict = dict()
        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        logits = self._classification_layer(embedded_text)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict.update({"logits": logits, "probs": probs})

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)

        return output_dict

    def get_embeddings(self, tokens: Union[TextFieldTensors, OneHot]) -> Dict[str, torch.Tensor]:
        if isinstance(tokens, OneHot):
            # TODO: sparse tensors support
            embedded_text = torch.matmul(tokens, self._text_field_embedder._token_embedders["tokens"].weight)
            indexes = torch.argmax(tokens, dim=-1)
            mask = (~torch.eq(indexes, 0)).float()
            token_ids = indexes
        else:
            token_ids = get_token_ids_from_text_field_tensors(tokens)
            embedded_text = self._text_field_embedder(tokens)
            mask = get_text_field_mask(tokens)

        return {"embedded_text": embedded_text, "mask": mask, "token_ids": token_ids}

    def forward(  # type: ignore
        self, tokens: Union[TextFieldTensors, OneHot], label: torch.IntTensor = None
    ) -> Dict[str, torch.Tensor]:

        emb_out = self.get_embeddings(tokens)

        output_dict = self.forward_on_embeddings(emb_out["embedded_text"], emb_out["mask"], label)
        output_dict["token_ids"] = emb_out["token_ids"]
        return output_dict
