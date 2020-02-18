from typing import List

import numpy as np
from allennlp.predictors import TextClassifierPredictor

from adat.models.classification_model import Classifier
from adat.dataset import ClassificationReader


class ClassifierPredictor:
    def __init__(self, model: Classifier) -> None:
        self.model = model
        self.reader = ClassificationReader()
        self.predictor = TextClassifierPredictor(self.model, self.reader)

    def predict(self, sequences: List[str]) -> np.ndarray:
        probs = [self.predictor.predict(seq)['probs'] for seq in sequences]
        probs = np.array(probs)
        return probs
