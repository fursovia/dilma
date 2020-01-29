from typing import Dict

import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Attention, Embedding
from allennlp.modules.attention import AdditiveAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import util

from adat.models import OneLanguageSeq2SeqModel


class OneLanguageSeq2SeqModelWithMasks(OneLanguageSeq2SeqModel):

    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 masker_embedder: TextFieldEmbedder,
                 max_decoding_steps: int,
                 attention: Attention,
                 beam_size: int = None
                 ) -> None:
        super().__init__(
            vocab=vocab,
            source_embedder=source_embedder,
            encoder=encoder,
            max_decoding_steps=max_decoding_steps,
            attention=attention,
            beam_size=beam_size,
        )
        self._masker_embedder = masker_embedder
        self._masker_aggregator = BagOfEmbeddingsEncoder(
            embedding_dim=self._masker_embedder.get_output_dim(),
            averaged=True
        )

    def _encode_masker(self, masker_tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        embedded_input = self._masker_embedder(masker_tokens)
        masker_mask = util.get_text_field_mask(masker_tokens)
        masker_hidden = self._masker_aggregator(embedded_input, masker_mask)
        return masker_hidden

    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        state = self._encode(source_tokens)
        # encode masks
        masker_tokens = kwargs.get('masker_tokens')  # Dict[str, torch.LongTensor]
        masker_hidden = self._encode_masker(masker_tokens)

        if target_tokens:
            state = self._init_decoder_state(state)
            state["decoder_hidden"].add_(masker_hidden)
            # The `_forward_loop` decodes the input sequence and computes the loss during training
            # and validation.
            output_dict = self._forward_loop(state, target_tokens)
        else:
            output_dict = {}

        if not self.training:
            state = self._init_decoder_state(state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)
            if target_tokens and self._bleu:
                # shape: (batch_size, beam_size, max_sequence_length)
                top_k_predictions = output_dict["predictions"]
                # shape: (batch_size, max_predicted_sequence_length)
                best_predictions = top_k_predictions[:, 0, :]
                self._bleu(best_predictions, target_tokens["tokens"])

        return output_dict


def get_mask_seq2seq_model(vocab: Vocabulary,
                           max_decoding_steps: int = 20,
                           beam_size: int = 1) -> OneLanguageSeq2SeqModel:
    emb_dim = 64
    hidden_dim = 32
    word_embeddings = Embedding(
        num_embeddings=vocab.get_vocab_size('tokens'),
        embedding_dim=emb_dim
    )
    word_embeddings = BasicTextFieldEmbedder({"tokens": word_embeddings})

    masker_embeddings = Embedding(
        num_embeddings=vocab.get_vocab_size('mask_tokens'),
        embedding_dim=hidden_dim
    )
    masker_embeddings = BasicTextFieldEmbedder({"tokens": masker_embeddings})

    attention = AdditiveAttention(vector_dim=hidden_dim, matrix_dim=hidden_dim)
    lstm = PytorchSeq2SeqWrapper(nn.LSTM(emb_dim, hidden_dim, batch_first=True))

    model = OneLanguageSeq2SeqModelWithMasks(
        vocab=vocab,
        source_embedder=word_embeddings,
        encoder=lstm,
        max_decoding_steps=max_decoding_steps,
        attention=attention,
        beam_size=beam_size,
        masker_embedder=masker_embeddings
    )

    return model
