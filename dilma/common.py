from dataclasses import dataclass
from dataclasses_json import dataclass_json


MASK_TOKEN = "@@MASK@@"
START_TOKEN = "@@START@@"
END_TOKEN = "@@END@@"


@dataclass_json
@dataclass
class SequenceData:
    sequence: str
    label: int
