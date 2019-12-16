import torch
from allennlp.models import Model


def load_weights(model: Model, path: str, location: str = 'cpu') -> None:
    with open(path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=location))
