from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.nn.modules.rnn import LSTMCell
from allennlp.data import Vocabulary
from allennlp.models import SimpleSeq2Seq
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Attention, Embedding
from allennlp.modules.attention import AdditiveAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn import util


class OneLanguageSeq2SeqModel(SimpleSeq2Seq):

    def get_state_for_beam_search(self,
                                  source_tokens: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        assert not self.training
        state = self._encode(source_tokens)
        # decoder_hidden should be modified
        state = self._init_decoder_state(state)
        return state

    def beam_search(self,
                    state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert not self.training
        predictions = self._forward_beam_search(state)
        return predictions

    def _prepare_output_projections(self,
                                    last_predictions: torch.Tensor,
                                    state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        encoder_outputs = state["encoder_outputs"]
        source_mask = state["source_mask"]
        decoder_hidden = state["decoder_hidden"]
        decoder_context = state["decoder_context"]
        embedded_input = self._source_embedder._token_embedders['tokens'](last_predictions)
        if self._attention:
            attended_input = self._prepare_attended_input(decoder_hidden, encoder_outputs, source_mask)
            decoder_input = torch.cat((attended_input, embedded_input), -1)
        else:
            decoder_input = embedded_input

        # hidden state and cell state
        decoder_hidden, decoder_context = self._decoder_cell(
                decoder_input,
                (decoder_hidden, decoder_context))

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context

        output_projections = self._output_projection_layer(decoder_hidden)
        return output_projections, state

    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        del kwargs
        return super().forward(source_tokens, target_tokens)


class OneLanguageSeq2SeqModelWithMasks(OneLanguageSeq2SeqModel):

    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 masker_embedder: TextFieldEmbedder,
                 max_decoding_steps: int,
                 attention: Attention = None,
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


def get_basic_seq2seq_model(vocab: Vocabulary, use_attention: bool = True,
                            max_decoding_steps: int = 20, beam_size: int = 1) -> OneLanguageSeq2SeqModel:
    emb_dim = 64
    hidden_dim = 32
    token_embedding = Embedding(
        num_embeddings=vocab.get_vocab_size('tokens'),
        embedding_dim=emb_dim
    )
    attention = AdditiveAttention(vector_dim=hidden_dim, matrix_dim=hidden_dim) if use_attention else None

    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    lstm = PytorchSeq2SeqWrapper(nn.LSTM(emb_dim, hidden_dim, batch_first=True))

    model = OneLanguageSeq2SeqModel(
        vocab=vocab,
        source_embedder=word_embeddings,
        encoder=lstm,
        max_decoding_steps=max_decoding_steps,
        attention=attention,
        beam_size=beam_size
    )

    return model


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
