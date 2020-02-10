from typing import Dict, Tuple

import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models import SimpleSeq2Seq
from allennlp.modules import Embedding
from allennlp.modules.attention import AdditiveAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder


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
        # TODO: pass state
        del kwargs
        return super().forward(source_tokens, target_tokens)


def get_seq2seq_model(vocab: Vocabulary,
                      max_decoding_steps: int = 20,
                      beam_size: int = 1, use_attention: bool = True) -> OneLanguageSeq2SeqModel:
    emb_dim = 64
    hidden_dim = 32
    token_embedding = Embedding(
        num_embeddings=vocab.get_vocab_size('tokens'),
        embedding_dim=emb_dim
    )
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    if use_attention:
        attention = AdditiveAttention(vector_dim=hidden_dim, matrix_dim=hidden_dim)
    else:
        attention = None
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
