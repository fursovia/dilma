from typing import Dict, Optional

import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.training.metrics import Perplexity
from allennlp.nn.util import get_text_field_mask
from allennlp_models.lm.language_model_heads import LinearLanguageModelHead

from adat.tokens_masker import TokensMasker


@Model.register("masked_lm")
class MaskedLanguageModel(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2seq_encoder: Seq2SeqEncoder,
        tokens_masker: Optional[TokensMasker] = None
    ) -> None:
        super().__init__(vocab)
        self._text_field_embedder = text_field_embedder
        self._seq2seq_encoder = seq2seq_encoder
        self._head = LinearLanguageModelHead(
            vocab=vocab,
            input_dim=self._seq2seq_encoder.get_output_dim(),
            vocab_namespace="tokens"
        )
        self._tokens_masker = tokens_masker

        ignore_index = -100 if self._tokens_masker is None else self._tokens_masker.skip_index
        self._loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self._perplexity = Perplexity()

    def forward(
        self,
        tokens: TextFieldTensors
    ) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)

        if self._tokens_masker is not None:
            tokens, targets = self._tokens_masker.mask_tokens(tokens)
        else:
            # TODO: pad-tokens in the loss
            targets = tokens

        embeddings = self._text_field_embedder(tokens)
        contextual_embeddings = self._seq2seq_encoder(embeddings, mask)

        logits = self._head(contextual_embeddings)

        output_dict = {
            "contextual_embeddings": contextual_embeddings,
            "logits": logits
        }

        output_dict["loss"] = self._loss(
            logits.view(-1, self.vocab.get_vocab_size()),
            # TODO: it is not always tokens-tokens
            targets["tokens"]["tokens"].view(-1)
        )
        self._perplexity(output_dict["loss"])
        return output_dict

    def get_metrics(self, reset: bool = False):
        return {"perplexity": self._perplexity.get_metric(reset=reset)}
