from typing import Dict, Optional

import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.modules import Embedding
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from adat.models.classification_model import BoWMaxAndMeanEncoder
from adat.models.seq2seq_model import OneLanguageSeq2SeqModel


class DeepLevenshtein(Model):
    """
    Idea from https://www.aclweb.org/anthology/P18-1186.pdf
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 seq2seq_encoder: Seq2SeqEncoder,
                 seq2vec_encoder: Seq2VecEncoder) -> None:
        super().__init__(vocab)
        self._text_field_embedder = text_field_embedder
        self._seq2seq_encoder = seq2seq_encoder
        self._seq2vec_encoder = seq2vec_encoder

        # self._loss = torch.nn.MSELoss()
        self._loss = torch.nn.L1Loss()
        self._cosine_sim = torch.nn.CosineSimilarity()

    def encode_sequence(self, sequence: Dict[str, torch.LongTensor]) -> torch.Tensor:
        embedded_sequence = self._text_field_embedder(sequence)
        mask = get_text_field_mask(sequence).float()
        embedded_sequence = self._seq2seq_encoder(embedded_sequence, mask=mask)
        embedded_sequence = self._seq2vec_encoder(embedded_sequence, mask=mask)
        return embedded_sequence

    def forward(self,
                sequence_a: Dict[str, torch.LongTensor],
                sequence_b: Dict[str, torch.LongTensor],
                similarity: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        embedded_sequence_a = self.encode_sequence(sequence_a)
        embedded_sequence_b = self.encode_sequence(sequence_b)

        cosine_similarity = self._cosine_sim(embedded_sequence_a, embedded_sequence_b)
        cosine_similarity_normalized = 0.5 * (cosine_similarity + 1)

        output_dict = {'normalized_cosine': cosine_similarity_normalized}

        if similarity is not None:
            loss = self._loss(cosine_similarity_normalized, similarity.view(-1))
            output_dict["loss"] = loss

        return output_dict


class DeepLevenshteinFromSeq2Seq(DeepLevenshtein):
    def __init__(self,
                 one_lang_seq2seq: OneLanguageSeq2SeqModel, seq2vec_encoder: Seq2VecEncoder) -> None:
        one_lang_seq2seq.eval()
        for p in one_lang_seq2seq.parameters():
            p.requires_grad = False

        super().__init__(
            vocab=one_lang_seq2seq.vocab,
            text_field_embedder=one_lang_seq2seq._source_embedder,
            seq2seq_encoder=one_lang_seq2seq._encoder,
            seq2vec_encoder=seq2vec_encoder
        )
        self._one_lang_seq2seq = one_lang_seq2seq


class DeepLevenshteinWithAttention(DeepLevenshtein):
    pass


def get_basic_deep_levenshtein(vocab: Vocabulary):
    embedding_dim = 64
    hidden_dim = 32

    token_embedding = Embedding(
        num_embeddings=vocab.get_vocab_size('tokens'),
        embedding_dim=embedding_dim
    )
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True))
    body = BoWMaxAndMeanEncoder(embedding_dim=hidden_dim * 2)

    model = DeepLevenshtein(
        vocab=vocab,
        text_field_embedder=word_embeddings,
        seq2seq_encoder=lstm,
        seq2vec_encoder=body,
    )
    return model
