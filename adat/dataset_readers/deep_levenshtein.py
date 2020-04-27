from typing import Optional, Dict
import json

import numpy as np
from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, Instance, Field
from allennlp.data.fields import TextField, ArrayField
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.token_indexers import TokenIndexer


@DatasetReader.register(name="deep_levenshtein")
class DeepLevenshteinReader(DatasetReader):
    def __init__(
            self,
            token_indexers: Dict[str, TokenIndexer],
            tokenizer: Tokenizer,
            lazy: bool = False
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers
        self._tokenizer = tokenizer

    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            for line in data_file.readlines():
                if not line:
                    continue
                items = json.loads(line)
                seq_a = items["seq_a"]
                seq_b = items["seq_b"]
                dist = items.get("dist")
                instance = self.text_to_instance(sequence_a=seq_a, sequence_b=seq_b, distance=dist)
                yield instance

    def text_to_instance(
        self,
        sequence_a: str,
        sequence_b: str,
        distance: Optional[float] = None
    ) -> Instance:
        fields: Dict[str, Field] = dict()
        fields["sequence_a"] = TextField(self._tokenizer.tokenize(sequence_a), self._token_indexers)
        fields["sequence_b"] = TextField(self._tokenizer.tokenize(sequence_b), self._token_indexers)

        if distance is not None:
            fields["distance"] = ArrayField(
                array=np.array([distance])
            )

        return Instance(fields)
