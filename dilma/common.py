from typing import Dict, Union
from dataclasses import dataclass

import torch
from dataclasses_json import dataclass_json
from allennlp.data import TextFieldTensors


MASK_TOKEN = "@@MASK@@"
START_TOKEN = "@@START@@"
END_TOKEN = "@@END@@"


@dataclass_json
@dataclass
class SequenceData:
    sequence: str
    label: int


ModelsInput = Dict[str, Union[TextFieldTensors, torch.Tensor]]
