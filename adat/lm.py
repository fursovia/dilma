

import torch.nn as nn
from allennlp.models.language_model import LanguageModel
from allennlp.models.masked_language_model import MaskedLanguageModel

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
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
