import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from allennlp.predictors import Predictor

from adat.utils import load_jsonlines

parser = argparse.ArgumentParser()
parser.add_argument("--adversarial-dir", type=str, required=True)
parser.add_argument("--classifier-dir", type=str, required=True)
parser.add_argument("--test-path", type=str, required=True)

parser.add_argument("--cuda", type=int, default=-1)


if __name__ == "__main__":
    args = parser.parse_args()
    adversarial_dir = Path(args.adversarial_dir)
    classifier_dir = Path(args.classifier_dir)
    test = load_jsonlines(args.test_path)

    labels = np.array([int(el['label']) for el in test])

    predictor = Predictor.from_path(
        classifier_dir / "model.tar.gz",
        predictor_name="text_classifier",
        cuda_device=args.cuda
    )

    preds = predictor.predict_batch_json([{"sentence": el["text"]} for el in test])
    probs = np.array([p['probs'] for p in preds])

    roc_auc = roc_auc_score(labels, probs[:, 1])
    print(">>>>>>> ROC AUC =", roc_auc)

    with open(classifier_dir / "discr_metrics.json", "w") as f:
        json.dump({"roc_auc": roc_auc}, f, indent=4)
