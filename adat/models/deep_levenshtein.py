from typing import Dict, Optional

import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn import util
from allennlp.modules import Embedding
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.attention import AdditiveAttention
from allennlp.modules import Attention

from adat.models.classification_model import BoWMaxAndMeanEncoder
from adat.models.masked_copynet import MaskedCopyNet


EMB_DIM = 64
HID_DIM = 32


class DeepLevenshtein(Model):
    """
    Idea from https://www.aclweb.org/anthology/P18-1186.pdf
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 seq2seq_encoder: Seq2SeqEncoder,
                 seq2vec_encoder: Seq2VecEncoder,
                 attention: Optional[Attention] = None) -> None:
        super().__init__(vocab)
        self.text_field_embedder = text_field_embedder
        self.seq2seq_encoder = seq2seq_encoder
        self.seq2vec_encoder = seq2vec_encoder
        self.attention = attention

        self._loss = torch.nn.L1Loss()
        self._cosine_sim = torch.nn.CosineSimilarity()

    def prepare_attended_input(
            self,
            seq_attention_from: Dict[str, torch.Tensor],
            seq_attention_to: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        input_weights = self.attention(
            seq_attention_from['vector'],
            seq_attention_to['matrix'],
            seq_attention_to['mask']
        )
        attended_input = util.weighted_sum(seq_attention_to['matrix'], input_weights)
        attented_seq = torch.cat((seq_attention_from['vector'], attended_input), -1)
        return attented_seq

    def calculate_similarity(
            self,
            embedded_sequence_a: Dict[str, torch.Tensor],
            embedded_sequence_b: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if self.attention:
            vector_a = self.prepare_attended_input(embedded_sequence_a, embedded_sequence_b)
            vector_b = self.prepare_attended_input(embedded_sequence_b, embedded_sequence_a)
        else:
            vector_a = embedded_sequence_a['vector']
            vector_b = embedded_sequence_b['vector']

        return 0.5 * (self._cosine_sim(vector_a, vector_b) + 1.0)

    def encode_sequence(self, sequence: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        embedded_sequence = self.text_field_embedder(sequence)
        mask = util.get_text_field_mask(sequence).float()
        embedded_sequence_matrix = self.seq2seq_encoder(embedded_sequence, mask=mask)
        embedded_sequence_vector = self.seq2vec_encoder(embedded_sequence_matrix, mask=mask)
        output = {'mask': mask, 'vector': embedded_sequence_vector, 'matrix': embedded_sequence_matrix}
        return output

    def forward(self,
                sequence_a: Dict[str, torch.LongTensor],
                sequence_b: Dict[str, torch.LongTensor],
                similarity: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        embedded_sequence_a = self.encode_sequence(sequence_a)
        embedded_sequence_b = self.encode_sequence(sequence_b)

        cosine_similarity_normalized = self.calculate_similarity(embedded_sequence_a, embedded_sequence_b)
        output_dict = {'normalized_cosine': cosine_similarity_normalized}

        if similarity is not None:
            loss = self._loss(cosine_similarity_normalized, similarity.view(-1))
            output_dict["loss"] = loss

        return output_dict


def get_deep_levenshtein(vocab: Vocabulary) -> DeepLevenshtein:
    token_embedding = Embedding(
        num_embeddings=vocab.get_vocab_size('tokens'),
        embedding_dim=EMB_DIM
    )
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMB_DIM, HID_DIM, batch_first=True, bidirectional=True))
    body = BoWMaxAndMeanEncoder(embedding_dim=HID_DIM * 2, hidden_dim=[64, 32])

    model = DeepLevenshtein(
        vocab=vocab,
        text_field_embedder=word_embeddings,
        seq2seq_encoder=lstm,
        seq2vec_encoder=body,
    )
    return model


def get_deep_levenshtein_attention(vocab: Vocabulary) -> DeepLevenshtein:

    token_embedding = Embedding(
        num_embeddings=vocab.get_vocab_size('tokens'),
        embedding_dim=EMB_DIM
    )
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMB_DIM, HID_DIM, batch_first=True, bidirectional=True))
    body = BoWMaxAndMeanEncoder(embedding_dim=HID_DIM * 2, hidden_dim=[64, 32])
    attention = AdditiveAttention(vector_dim=body.get_output_dim(), matrix_dim=HID_DIM * 2)

    model = DeepLevenshtein(
        vocab=vocab,
        text_field_embedder=word_embeddings,
        seq2seq_encoder=lstm,
        seq2vec_encoder=body,
        attention=attention
    )
    return model


def get_deep_levenshtein_copynet(masked_copynet: MaskedCopyNet) -> DeepLevenshtein:
    masked_copynet.eval()
    for p in masked_copynet.parameters():
        p.requires_grad = False

    hidden_dim = masked_copynet._encoder_output_dim
    body = BoWMaxAndMeanEncoder(embedding_dim=hidden_dim, hidden_dim=[64, 32])
    attention = AdditiveAttention(vector_dim=body.get_output_dim(), matrix_dim=HID_DIM * 2)

    model = DeepLevenshtein(
        vocab=masked_copynet.vocab,
        text_field_embedder=masked_copynet._embedder,
        seq2seq_encoder=masked_copynet._encoder,
        seq2vec_encoder=body,
        attention=attention
    )
    return model
