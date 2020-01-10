from typing import Dict, Tuple

import torch
import torch.nn as nn
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.models.language_model import LanguageModel
from allennlp.models.basic_classifier import BasicClassifier
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.modules.attention.additive_attention import AdditiveAttention
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.modules.token_embedders import Embedding


def get_basic_lm(vocab: Vocabulary) -> LanguageModel:
    emb_dim = 64
    hidden_dim = 32
    token_embedding = Embedding(
        num_embeddings=vocab.get_vocab_size('tokens'),
        embedding_dim=emb_dim
    )

    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    lstm = PytorchSeq2SeqWrapper(nn.LSTM(emb_dim, hidden_dim, batch_first=True))
    model = LanguageModel(
        vocab=vocab,
        text_field_embedder=word_embeddings,
        contextualizer=lstm
    )

    return model


def get_basic_classification_model(vocab: Vocabulary) -> BasicClassifier:
    embedding_dim = 32
    hidden_dim = 16

    token_embedding = Embedding(
        num_embeddings=vocab.get_vocab_size('tokens'),
        embedding_dim=embedding_dim
    )

    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True))
    body = BagOfEmbeddingsEncoder(embedding_dim=hidden_dim)
    model = BasicClassifier(
        vocab=vocab,
        text_field_embedder=word_embeddings,
        seq2seq_encoder=lstm,
        seq2vec_encoder=body,
        num_labels=2
    )
    return model


class OneLanguageSeq2SeqModel(SimpleSeq2Seq):

    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 max_decoding_steps: int,
                 attention: Attention = None,
                 attention_function: SimilarityFunction = None,
                 beam_size: int = None,
                 target_namespace: str = "tokens",
                 target_embedding_dim: int = None,
                 scheduled_sampling_ratio: float = 0.,
                 use_bleu: bool = True) -> None:
        super().__init__(
            vocab,
            source_embedder,
            encoder,
            max_decoding_steps,
            attention,
            attention_function,
            beam_size,
            target_namespace,
            target_embedding_dim,
            scheduled_sampling_ratio,
            use_bleu
        )

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


def get_basic_seq2seq_model(vocab: Vocabulary) -> SimpleSeq2Seq:
    emb_dim = 64
    hidden_dim = 32
    token_embedding = Embedding(
        num_embeddings=vocab.get_vocab_size('tokens'),
        embedding_dim=emb_dim
    )

    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    lstm = PytorchSeq2SeqWrapper(nn.LSTM(emb_dim, hidden_dim, batch_first=True))

    model = OneLanguageSeq2SeqModel(
        vocab=vocab,
        source_embedder=word_embeddings,
        encoder=lstm,
        max_decoding_steps=20,
        attention=AdditiveAttention(vector_dim=hidden_dim, matrix_dim=hidden_dim)
    )

    return model
