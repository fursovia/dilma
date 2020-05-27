from typing import Dict
import json

from allennlp_models.lm import SimpleLanguageModelingDatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
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

    def text_to_instance(
        self,  # type: ignore
        sentence: str = None,
        sequence: str = None
    ) -> Instance:

        inp = sentence or sequence
        tokenized = self._tokenizer.tokenize(inp)
        tokenized_with_ends = []
        tokenized_with_ends.extend(self._start_tokens)
        tokenized_with_ends.extend(tokenized)
        tokenized_with_ends.extend(self._end_tokens)
        return_instance = Instance({"source": TextField(tokenized_with_ends, self._token_indexers)})
        return return_instance

    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            for line in data_file.readlines():
                if not line:
                    continue
                items = json.loads(line)
                sent = items.get("text")
                seq = items.get("sequence")
                instance = self.text_to_instance(sequence=seq, sentence=sent)
                yield instance
