from typing import Dict, Optional

import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp_models.lm.modules import LinearLanguageModelHead
from allennlp.training.metrics import Perplexity

from dilma.modules.masker import TokensMasker


@Model.register("masked_lm")
class MaskedLanguageModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        sequence_field_embedder: TextFieldEmbedder,
        seq2seq_encoder: Seq2SeqEncoder,
        tokens_masker: Optional[TokensMasker] = None,
    ) -> None:
        super().__init__(vocab)
        self._sequence_field_embedder = sequence_field_embedder
        self._seq2seq_encoder = seq2seq_encoder
        self._head = LinearLanguageModelHead(
            vocab=vocab, input_dim=self._seq2seq_encoder.get_output_dim(), vocab_namespace="tokens"
        )
        self._tokens_masker = tokens_masker

        ignore_index = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN)
        self._loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self._perplexity = Perplexity()

    def get_output_dim(self) -> int:
        return self._seq2seq_encoder.get_output_dim()

    def forward(
        self, sequence: TextFieldTensors, **kwargs,
    ) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sequence)

        if self._tokens_masker is not None:
            sequence, targets = self._tokens_masker.mask_tokens(sequence)
        else:
            targets = None

        sequence_embeddings = self._sequence_field_embedder(sequence)
        contextual_embeddings = self._seq2seq_encoder(sequence_embeddings, mask)

        logits = self._head(contextual_embeddings)

        output_dict = dict(contextual_embeddings=contextual_embeddings, logits=logits, mask=mask)

        if targets is not None:
            output_dict["loss"] = self._loss(
                logits.transpose(1, 2),
                # TODO: it is not always tokens-tokens
                targets["tokens"]["tokens"],
            )
            self._perplexity(output_dict["loss"])
        return output_dict

    def get_metrics(self, reset: bool = False):
        return {"perplexity": self._perplexity.get_metric(reset=reset)}
