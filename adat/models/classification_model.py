import torch
from allennlp.data import Vocabulary
from allennlp.models import BasicClassifier
from allennlp.modules import Embedding
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder


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
