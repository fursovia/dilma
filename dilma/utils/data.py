from typing import Union, List, Dict, Any, Sequence

import torch
import jsonlines
from allennlp.data import Batch
from allennlp.nn.util import move_to_device
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.vocabulary import Vocabulary

from dilma.common import SequenceData, ModelsInput


def data_to_tensors(
    data: SequenceData, reader: DatasetReader, vocab: Vocabulary, device: Union[torch.device, int] = -1
) -> ModelsInput:

    instances = Batch([reader.text_to_instance(**data.to_dict())])

    instances.index_instances(vocab)
    inputs = instances.as_tensor_dict()
    return move_to_device(inputs, device)


def decode_indexes(
    indexes: torch.Tensor, vocab: Vocabulary, namespace="transactions", drop_start_end: bool = True,
) -> str:
    out = [vocab.get_token_from_index(idx.item(), namespace=namespace) for idx in indexes]

    if drop_start_end:
        out = out[1:-1]

    return " ".join(out)


def load_jsonlines(path: str) -> List[Dict[str, Any]]:
    data = []
    with jsonlines.open(path, "r") as reader:
        for items in reader:
            data.append(items)
    return data


def write_jsonlines(data: Sequence[Dict[str, Any]], path: str) -> None:
    with jsonlines.open(path, "w") as writer:
        for ex in data:
            writer.write(ex)
