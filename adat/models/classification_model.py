from typing import List, Tuple

import torch
import numpy as np
from allennlp.data import Vocabulary
from allennlp.models import BasicClassifier
from allennlp.modules import Embedding
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def get_basic_classification_model(vocab: Vocabulary, num_classes: int = 2) -> BasicClassifier:
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
        num_labels=num_classes
    )
    return model


class LogisticRegressionOnTfIdf:
    def __init__(self, ngram_range: Tuple[int, int] = (1, 2)) -> None:
        self.model = LogisticRegression()
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
