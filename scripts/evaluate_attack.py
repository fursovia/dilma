import argparse
from pathlib import Path
from typing import List
import json

import numpy as np
from allennlp.predictors import Predictor

from adat.utils import load_jsonlines, calculate_wer

parser = argparse.ArgumentParser()
parser.add_argument("test-path", type=str, required=True)
parser.add_argument("adversarial-test-path", type=str, required=True)
parser.add_argument("target-model-dir", type=str, required=True)
parser.add_argument("out-dir", type=str, required=True)
parser.add_argument("sample-size", type=int, default=None)


def normalized_accuracy_drop(
        wers: List[int],
        y_true: List[int],
        y_adv: List[int]
) -> float:
    assert len(y_true) == len(y_adv)
    nads = []
    for wer, lab, alab in zip(wers, y_true, y_adv):
        if wer > 0 and lab != alab:
            nads.append(1 / wer)
        else:
            nads.append(0.0)

    return sum(nads) / len(nads)


def calculate_wers(
        sequences: List[str],
        adv_sequences: List[str],
) -> List[int]:
    wers = []
    for seq, aseq in zip(sequences, adv_sequences):
        wer = calculate_wer(seq, aseq)
        wers.append(wer)
    return wers


if __name__ == "__main__":
    args = parser.parse_args()
    target_model_dir = Path(args.target_model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    test = load_jsonlines(args.test_path)[:args.sample_size]
    adversarial_test = load_jsonlines(args.adversarial_test_path)[:args.sample_size]

    predictor = Predictor.from_path(
        target_model_dir / "model.tar.gz",
        predictor_name="text_classifier"
    )
    preds = predictor.predict_batch_json(test)
    y_true = [int(el["label"]) for el in test]
    adv_preds = predictor.predict_batch_json(adversarial_test)
    y_adv = [int(el["label"]) for el in adv_preds]

    prob_diffs = [p["probs"][l] - ap["probs"][l] for p, ap, l in zip(preds, adv_preds, y_true)]

    wers = calculate_wers(
        sequences=[el['text'] for el in test],
        adv_sequences=[el['text'] for el in adversarial_test],
    )

    nad = normalized_accuracy_drop(
        wers=wers,
        y_true=y_true,
        y_adv=y_adv
    )

    metrics = dict(
        mean_prob_diff=float(np.mean(prob_diffs)),
        mean_wer=float(np.mean(wers)),
        NAD=nad
    )

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
