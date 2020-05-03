import argparse
from pathlib import Path
from typing import List
from pprint import pprint
import json

import numpy as np
from allennlp.predictors import Predictor

from adat.utils import load_jsonlines

parser = argparse.ArgumentParser()
parser.add_argument("--adversarial-dir", type=str, required=True)
parser.add_argument("--classifier-dir", type=str, required=True)
parser.add_argument("--gamma", type=float, default=1.0)


def normalized_accuracy_drop(
        wers: List[int],
        y_true: List[int],
        y_adv: List[int],
        gamma: float = 1.0
) -> float:
    assert len(y_true) == len(y_adv)
    nads = []
    for wer, lab, alab in zip(wers, y_true, y_adv):
        if wer > 0 and lab != alab:
            nads.append(1 / wer ** gamma)
        else:
            nads.append(0.0)

    return sum(nads) / len(nads)


if __name__ == "__main__":
    args = parser.parse_args()
    adversarial_dir = Path(args.adversarial_dir)
    data = load_jsonlines(adversarial_dir / "attacked_data.json")

    predictor = Predictor.from_path(
        Path(args.classifier_dir) / "model.tar.gz",
        predictor_name="text_classifier"
    )
    preds = predictor.predict_batch_json([{"sentence": el["sequence"]} for el in data])
    y_true = [int(el["attacked_label"]) for el in data]
    correctly_predicted = [y == int(p["label"]) for y, p in zip(y_true, preds)]

    adv_preds = predictor.predict_batch_json([{"sentence": el["adversarial_sequence"]} for el in data])
    y_adv = [int(el["label"]) for el in adv_preds]

    prob_diffs = [p["probs"][l] - ap["probs"][l] for p, ap, l in zip(preds, adv_preds, y_true)]
    wers = [el["wer"] for el in data]

    nad = normalized_accuracy_drop(
        wers=wers,
        y_true=y_true,
        y_adv=y_adv,
        gamma=args.gamma
    )

    metrics = dict(
        mean_prob_diff=float(np.mean(prob_diffs)),
        mean_wer=float(np.mean(wers)),
    )
    metrics[f"NAD_{args.gamma}"] = nad

    pprint(metrics)
    with open(adversarial_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
