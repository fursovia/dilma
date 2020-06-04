import argparse
from glob import glob
from pathlib import Path
import json
import re

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", type=str, default="./results")
parser.add_argument("--adv-training", action="store_true")

METRIC_NAMES = ["NAD_1.0", "mean_prob_diff", "mean_wer", "misclassification_error"]


if __name__ == "__main__":
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    metrics_df = []
    if not args.adv_training:
        for path in glob(str(results_dir / "*/*/*")):
            path = Path(path)
            with open(path / "target_clf_metrics.json") as f:
                metrics = json.load(f)

            metrics = {k: round(v, 2) for k, v in metrics.items() if k in METRIC_NAMES}
            metrics["dataset"] = path.parent.name
            metrics["adversary"] = path.name
            metrics_df.append(metrics)
    else:
        for path in glob(str(results_dir / "*/*/*/*0_metrics.json")):
            path = Path(path)
            with open(path) as f:
                metrics = json.load(f)

            metrics = {k: round(v, 2) for k, v in metrics.items() if k in METRIC_NAMES}
            metrics["dataset"] = path.parent.parent.name
            metrics["adversary"] = path.parent.name
            metrics["num_adv_examples"] = int(re.sub(r"[^\d]+", "", path.name))
            metrics_df.append(metrics)

    metrics_df = pd.DataFrame(metrics_df)
    # change the order of columns
    metrics_df = metrics_df[["dataset", "adversary"] + METRIC_NAMES]
    metrics_df.to_csv(results_dir / "results.csv", index=False)
    print(metrics_df.to_markdown())
