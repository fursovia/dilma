from typing import Iterator, List

from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer


class WhitespaceTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[Token]:
        return [Token(t) for t in text.split()]


class TexarInsuranceReader(DatasetReader):
    def text_to_instance(self, sentence: str, label: int = None) -> Instance:
        if not isinstance(sentence, list):
            sentence = sentence.split()

        sentence_field = TextField([Token(word) for word in sentence], {"tokens": SingleIdTokenIndexer()})
        fields = {"tokens": sentence_field}

        if label is not None:
            label_field = LabelField(label=label, skip_indexing=True)
            fields["label"] = label_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        text_path = file_path + '.text'
        labels_path = file_path + '.labels'

        with open(text_path) as text_f, open(labels_path) as labels_f:
            for line_t, line_l in zip(text_f, labels_f):
                sentence = line_t.strip()
                label = int(line_l.strip())
                yield self.text_to_instance(sentence, label)
