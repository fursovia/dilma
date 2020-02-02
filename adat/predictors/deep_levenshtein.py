from copy import deepcopy
from typing import List, Dict

import numpy as np
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import ArrayField


@Predictor.register('deep_levenshtein')
class DeepLevenshteinPredictor(Predictor):
    def predict(self, sequence_a: str, sequence_b: str) -> JsonDict:
        return self.predict_json({"sequence_a": sequence_a, "sequence_b": sequence_b})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"label"`` to the output.
        """
        sequence_a = json_dict["sequence_a"]
        sequence_b = json_dict["sequence_b"]
        return self._dataset_reader.text_to_instance(sequence_a, sequence_b)

    def predictions_to_labeled_instances(self,
                                         instance: Instance,
                                         outputs: Dict[str, np.ndarray]) -> List[Instance]:
        new_instance = deepcopy(instance)
        normalized_cosine = outputs['normalized_cosine']
        new_instance.add_field('similarity', ArrayField(array=np.array([normalized_cosine])))
        return [new_instance]
