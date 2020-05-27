from typing import Dict

from allennlp_models.lm import SimpleLanguageModelingDatasetReader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers.tokenizer import Tokenizer


@DatasetReader.register("simple_language_modeling_fixed")
class SimpleLanguageModelingDatasetReaderFixed(SimpleLanguageModelingDatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_sequence_length: int = None,
        lazy: bool = False
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            token_indexers=token_indexers,
            max_sequence_length=max_sequence_length,
            start_tokens=["<START>"],
            end_tokens=["<END>"],
            lazy=lazy
        )
