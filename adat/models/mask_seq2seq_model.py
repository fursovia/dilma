from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.nn.modules.rnn import LSTMCell
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Attention, Embedding
from allennlp.modules.attention import AdditiveAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import util

from adat.models import OneLanguageSeq2SeqModel
from adat.dataset import IDENTITY_TOKEN


class OneLanguageSeq2SeqModelWithAttMasks(OneLanguageSeq2SeqModel):

    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 masker_embedder: TextFieldEmbedder,
                 masker_attention: Attention,
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
        self._masker_attention = masker_attention
        self._decoder_input_dim += self._masker_embedder.get_output_dim()
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)

    def _prepare_mask_attended_input(self,
                                     decoder_hidden_state: torch.LongTensor = None,
                                     masker_encoder_outputs: torch.LongTensor = None,
                                     masker_encoder_outputs_mask: torch.LongTensor = None) -> torch.Tensor:
        encoder_outputs_mask = masker_encoder_outputs_mask.float()
        input_weights = self._masker_attention(
                decoder_hidden_state, masker_encoder_outputs, encoder_outputs_mask)
        attended_input = util.weighted_sum(masker_encoder_outputs, input_weights)
        return attended_input

    def _prepare_output_projections(self,
                                    last_predictions: torch.Tensor,
                                    state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        encoder_outputs = state["encoder_outputs"]
        source_mask = state["source_mask"]

        masker_encoder_outputs = state["masker_encoder_outputs"]
        masker_source_mask = state["masker_source_mask"]

        decoder_hidden = state["decoder_hidden"]
        decoder_context = state["decoder_context"]
        embedded_input = self._source_embedder._token_embedders['tokens'](last_predictions)
        if self._attention:
            attended_input = self._prepare_attended_input(decoder_hidden, encoder_outputs, source_mask)
            masker_attended_input = self._prepare_mask_attended_input(
                decoder_hidden,
                masker_encoder_outputs,
                masker_source_mask
            )
            decoder_input = torch.cat((attended_input, masker_attended_input, embedded_input), -1)
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

    def _encode_masker(self, masker_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        embedded_input = self._masker_embedder(masker_tokens)
        masker_mask = util.get_text_field_mask(masker_tokens)
        return {
                "masker_source_mask": masker_mask,
                "masker_encoder_outputs": embedded_input,
        }

    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        batch_size = source_tokens['tokens'].shape[0]
        state = self._encode(source_tokens) if 'state' not in kwargs else kwargs.get('state')
        if 'masker_tokens' in kwargs:
            masker_tokens = kwargs.get('masker_tokens')  # Dict[str, torch.LongTensor]
        else:
            masker_tokens = {
                "tokens": torch.tensor(
                    [self.vocab.get_token_index(IDENTITY_TOKEN, 'mask_tokens')] * batch_size,
                    device=source_tokens['tokens'].device
                ).reshape(-1, 1)
            }
        # encode masks
        state.update(self._encode_masker(masker_tokens))

        if target_tokens:
            state = self._init_decoder_state(state)
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


def get_att_mask_seq2seq_model(vocab: Vocabulary,
                               max_decoding_steps: int = 20,
                               beam_size: int = 1) -> OneLanguageSeq2SeqModel:
    emb_dim = 64
    hidden_dim = 32
    word_embeddings = Embedding(
        num_embeddings=vocab.get_vocab_size('tokens'),
        embedding_dim=emb_dim
    )
    word_embeddings = BasicTextFieldEmbedder({"tokens": word_embeddings})

    masker_emb_dim = 16
    masker_embeddings = Embedding(
        num_embeddings=vocab.get_vocab_size('mask_tokens'),
        embedding_dim=masker_emb_dim
    )
    masker_embeddings = BasicTextFieldEmbedder({"tokens": masker_embeddings})

    attention = AdditiveAttention(vector_dim=hidden_dim, matrix_dim=hidden_dim)
    masker_attention = AdditiveAttention(vector_dim=hidden_dim, matrix_dim=masker_emb_dim)
    lstm = PytorchSeq2SeqWrapper(nn.LSTM(emb_dim, hidden_dim, batch_first=True))

    model = OneLanguageSeq2SeqModelWithAttMasks(
        vocab=vocab,
        source_embedder=word_embeddings,
        encoder=lstm,
        max_decoding_steps=max_decoding_steps,
        attention=attention,
        beam_size=beam_size,
        masker_embedder=masker_embeddings,
        masker_attention=masker_attention
    )

    return model
