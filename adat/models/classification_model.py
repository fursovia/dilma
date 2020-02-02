from typing import List, Tuple, Dict

import torch
import numpy as np
from allennlp.data import Vocabulary
from allennlp.models import BasicClassifier
from allennlp.modules import Embedding
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.training.metrics import Auc, F1Measure
from allennlp.nn.util import get_text_field_mask
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class BoWMaxEncoder(Seq2VecEncoder):
    def __init__(self,
                 embedding_dim: int) -> None:
        super(BoWMaxEncoder, self).__init__()
        self._embedding_dim = embedding_dim

    def get_input_dim(self) -> int:
        return self._embedding_dim

    def get_output_dim(self) -> int:
        return self._embedding_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()

        argmaxed = tokens.max(dim=1).values
        return argmaxed


class BoWMaxAndMeanEncoder(Seq2VecEncoder):
    def __init__(self,
                 embedding_dim: int) -> None:
        super(BoWMaxAndMeanEncoder, self).__init__()
        self._embedding_dim = embedding_dim
        self.maxer = BoWMaxEncoder(self._embedding_dim)
        self.meaner = BagOfEmbeddingsEncoder(self._embedding_dim, True)

    def get_input_dim(self) -> int:
        return self._embedding_dim

    def get_output_dim(self) -> int:
        return self._embedding_dim * 2

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        argmaxed = self.maxer(tokens, mask)
        summed = self.meaner(tokens, mask)
        aggregated = torch.cat([argmaxed, summed], dim=1)
        return aggregated


class BasicClassifierWithMetric(BasicClassifier):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 seq2vec_encoder: Seq2VecEncoder,
                 seq2seq_encoder: Seq2SeqEncoder = None,
                 num_labels: int = None) -> None:

        super().__init__(vocab, text_field_embedder, seq2vec_encoder, seq2seq_encoder, num_labels=num_labels)
        if num_labels == 2:
            self._auc = Auc()
            self._f1 = F1Measure(1)

    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens).float()
        # TODO: hotflip bug
        # mask = None

        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        logits = self._classification_layer(embedded_text)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)

        if label is not None and self._num_labels == 2:
            self._auc(output_dict['probs'][:, 1], label.long().view(-1))
            self._f1(output_dict['probs'], label.long().view(-1))

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = super().get_metrics(reset)
        metrics.update({'auc': self._auc.get_metric(reset), 'f1': self._f1.get_metric(reset)[2]})
        return metrics


class BasicClassifierFromSeq2Seq(BasicClassifierWithMetric):
    pass


def get_basic_classification_model(vocab: Vocabulary, num_classes: int = 2) -> BasicClassifier:
    embedding_dim = 64
    hidden_dim = 32

    token_embedding = Embedding(
        num_embeddings=vocab.get_vocab_size('tokens'),
        embedding_dim=embedding_dim
    )

    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True))
    # body = BagOfEmbeddingsEncoder(embedding_dim=hidden_dim, averaged=True)
    body = BoWMaxAndMeanEncoder(embedding_dim=hidden_dim * 2)
    model = BasicClassifierWithMetric(
        vocab=vocab,
        text_field_embedder=word_embeddings,
        seq2seq_encoder=lstm,
        seq2vec_encoder=body,
        num_labels=num_classes
    )
    return model


class LogisticRegressionOnTfIdf:
    def __init__(self, ngram_range: Tuple[int, int] = (1, 2)) -> None:
        self.model = LogisticRegression(max_iter=10000, n_jobs=-1)
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range)

    def fit(self, sequences: List[str], labels: List[int]) -> 'LogisticRegressionOnTfIdf':
        train = self.vectorizer.fit_transform(sequences)
        self.model.fit(train, labels)
        return self

    def predict(self, sequences: List[str]) -> np.ndarray:
        data = self.vectorizer.transform(sequences)
        predictions = self.model.predict_proba(data)
        return predictions
