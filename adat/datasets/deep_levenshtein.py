import csv
from typing import Optional, Dict

import numpy as np
from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, Instance, Field
from allennlp.data.fields import TextField, ArrayField
from allennlp.data.tokenizers import WhitespaceTokenizer


class LevenshteinReader(DatasetReader):
    def __init__(self, lazy: bool = False):
        super().__init__(lazy)
        self._tokenizer = WhitespaceTokenizer()

    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            tsv_in = csv.reader(data_file, delimiter=',')
            next(tsv_in, None)
            for row in tsv_in:
                if len(row) == 3:
                    yield self.text_to_instance(sequence_a=row[0], sequence_b=row[1], similarity=row[2])
                else:
                    yield self.text_to_instance(sequence_a=row[0], sequence_b=row[1])

    def text_to_instance(
        self,
        sequence_a: str,
        sequence_b: str,
        similarity: Optional[float] = None
    ) -> Instance:
        fields: Dict[str, Field] = dict()
        fields["sequence_a"] = TextField(
            self._tokenizer.tokenize(sequence_a),
            {
                "tokens": _get_default_indexer()
            }
        )

        fields["sequence_b"] = TextField(
            self._tokenizer.tokenize(sequence_b),
            {
                "tokens": _get_default_indexer()
            }
        )

        if similarity is not None:
            # TODO: fix this hack
            fields["similarity"] = ArrayField(
                array=np.array([similarity])
            )

        return Instance(fields)
