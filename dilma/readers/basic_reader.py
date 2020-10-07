from typing import List, Optional
import jsonlines

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, LabelField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer, Token
from allennlp.models.archival import load_archive
import numpy as np

from dilma.common import START_TOKEN, END_TOKEN


@DatasetReader.register("basic_reader")
class BasicDatasetReader(DatasetReader):
    def __init__(self, lazy: bool = False,) -> None:
        super().__init__(lazy=lazy)

        self._tokenizer = WhitespaceTokenizer()
        self._start_token = Token(START_TOKEN)
        self._end_token = Token(END_TOKEN)

    def _add_start_end_tokens(self, tokens: List[Token]) -> List[Token]:
        return [self._start_token] + tokens + [self._end_token]

    def text_to_instance(
        self,
        sequence: Optional[str] = None,
        sequence_a: Optional[str] = None,
        sequence_b: Optional[str] = None,
        label: Optional[int] = None,
        distance: Optional[float] = None,
    ) -> Instance:

        fields = dict()

        if sequence is not None:
            tokens = self._tokenizer.tokenize(sequence)
            fields['sequence'] = TextField(tokens, {"tokens": SingleIdTokenIndexer()})

        if sequence_a is not None and sequence_b is not None:
            tokens_a = self._tokenizer.tokenize(sequence_a)
            fields["sequence_a"] = TextField(tokens_a, {"tokens": SingleIdTokenIndexer()})

            tokens_b = self._tokenizer.tokenize(sequence_b)
            fields["sequence_b"] = TextField(tokens_b, {"tokens": SingleIdTokenIndexer()})

        if label is not None:
            fields["label"] = LabelField(label=str(label), skip_indexing=False)

        if distance is not None:
            fields["distance"] = ArrayField(array=np.array(distance))

        return Instance(fields)

    def _read(self, file_path: str):

        with jsonlines.open(cached_path(file_path), "r") as reader:
            for items in reader:
                instance = self.text_to_instance(
                    sequence=items.get("sequence"),
                    sequence_a=items.get("sequence_a"),
                    sequence_b=items.get("sequence_b"),
                    label=items.get("label"),
                    distance=items.get("distance"),
                )
                yield instance

    @classmethod
    def from_archive(cls, archive_file: str) -> "BasicDatasetReader":
        config = load_archive(archive_file).config["dataset_reader"]
        assert config.pop("type") == "basic_reader"
        return cls(**config)


BasicDatasetReader.register("from_archive", constructor="from_archive")(BasicDatasetReader)
