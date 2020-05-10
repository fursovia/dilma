from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.modules.seq2vec_encoders import CnnEncoder, Seq2VecEncoder


@Seq2VecEncoder.register("distribution_cnn")
class DistributionCnnEncoder(CnnEncoder):
    @classmethod
    def from_params(cls, params: Params, vocab: Vocabulary, **extras) -> "DistributionCnnEncoder":
        embedding_dim = params.pop_int("embedding_dim", vocab.get_vocab_size("tokens"))
        num_filters = params.pop_int("num_filters")
        ngram_filter_sizes = params.pop("ngram_filter_sizes", (2, 3, 4, 5))

        params.assert_empty(cls.__name__)

        return cls(embedding_dim=embedding_dim, num_filters=num_filters, ngram_filter_sizes=ngram_filter_sizes)
