from typing import List, Tuple, Dict, Optional

import torch
import numpy as np
from allennlp.data import Vocabulary
from allennlp.models import BasicClassifier
from allennlp.modules import Embedding
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.training.metrics import Auc, F1Measure, CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from adat.models.masked_copynet import MaskedCopyNet


EMB_DIM = 64
HID_DIM = 32


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
                 embedding_dim: int, hidden_dim: Optional[List[int]] = None) -> None:
        super(BoWMaxAndMeanEncoder, self).__init__()
        self._embedding_dim = embedding_dim
        self.maxer = BoWMaxEncoder(self._embedding_dim)
        self.meaner = BagOfEmbeddingsEncoder(self._embedding_dim, True)
        self._hidden_dim = hidden_dim
        if self._hidden_dim is not None:
            layers = [
                torch.nn.LeakyReLU(),
                torch.nn.Linear(self._embedding_dim * 2, self._hidden_dim[0])
            ]

            for i, hid_dim in enumerate(self._hidden_dim[1:]):
                layers.append(torch.nn.LeakyReLU())
                layers.append(torch.nn.Linear(self._hidden_dim[i], hid_dim))

            self.linear = torch.nn.Sequential(*layers)
        else:
            self.linear = None

    def get_input_dim(self) -> int:
        return self._embedding_dim

    def get_output_dim(self) -> int:
        return self._embedding_dim * 2 if self.linear is None else self._hidden_dim[-1]

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        argmaxed = self.maxer(tokens, mask)
        summed = self.meaner(tokens, mask)
        output = torch.cat([argmaxed, summed], dim=1)
        if self.linear is not None:
            output = self.linear(output)
        return output


class Classifier(BasicClassifier):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 seq2vec_encoder: Seq2VecEncoder,
                 seq2seq_encoder: Seq2SeqEncoder = None,
                 num_labels: int = None) -> None:

        super().__init__(vocab, text_field_embedder, seq2vec_encoder, seq2seq_encoder, num_labels=num_labels)
        if self._num_labels == 2:
            self._auc = Auc()
            self._f1 = F1Measure(1)

    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens).float()
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

        if label is not None:
            if self._num_labels == 2:
                self._auc(output_dict['probs'][:, 1], label.long().view(-1))
                self._f1(output_dict['probs'], label.long().view(-1))
            
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = super().get_metrics(reset)
        if self._num_labels == 2:
            metrics.update({'auc': self._auc.get_metric(reset), 'f1': self._f1.get_metric(reset)[2]})
        return metrics


def get_classification_model(vocab: Vocabulary, num_classes: int = 2) -> BasicClassifier:

    token_embedding = Embedding(
        num_embeddings=vocab.get_vocab_size('tokens'),
        embedding_dim=EMB_DIM
    )

    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMB_DIM, HID_DIM, batch_first=True, bidirectional=True))
    body = BoWMaxAndMeanEncoder(embedding_dim=HID_DIM * 2)
    model = Classifier(
        vocab=vocab,
        text_field_embedder=word_embeddings,
        seq2seq_encoder=lstm,
        seq2vec_encoder=body,
        num_labels=num_classes
    )
    return model


def get_classification_model_copynet(
        masked_copynet: MaskedCopyNet,
        num_classes: int = 2
) -> BasicClassifier:
    masked_copynet.eval()
    for p in masked_copynet.parameters():
        p.requires_grad = False

    hidden_dim = masked_copynet._encoder_output_dim
    body = BoWMaxAndMeanEncoder(embedding_dim=hidden_dim, hidden_dim=[64, 32])
    model = Classifier(
        vocab=masked_copynet.vocab,
        text_field_embedder=masked_copynet._embedder,
        seq2seq_encoder=masked_copynet._encoder,
        seq2vec_encoder=body,
        num_labels=num_classes
    )
    return model


def get_logistic_regression(ngram_range: Tuple[int, int] = (1, 2)) -> Pipeline:
    pipeline = Pipeline(
        [
            ('tfidf', TfidfVectorizer(ngram_range=ngram_range)),
            ('logreg', LogisticRegression(max_iter=10000, n_jobs=-1))
        ]
    )
    return pipeline


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
