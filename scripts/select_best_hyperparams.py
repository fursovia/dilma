import argparse
from glob import glob
from pathlib import Path
import json

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", type=str, default="./results")
parser.add_argument("--alg-name", type=str, default="cascada")

METRIC_NAMES = ["NAD_1.0", "mean_prob_diff", "mean_wer"]


if __name__ == "__main__":
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    metrics_df = []
    for path in glob(str(results_dir / f"*/*/{args.alg_name}/grid_search/*")):
        path = Path(path)
        metrics_path = path / "target_clf_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            metrics = {k: round(v, 3) for k, v in metrics.items() if k in METRIC_NAMES}
        else:
            continue
        metrics["config_num"] = path.name
        metrics["dataset"] = path.parent.parent.parent.name
        metrics_df.append(metrics)

    metrics_df = pd.DataFrame(metrics_df)
    print("Print best hyperparams num: ", metrics_df.iloc[metrics_df["NAD_1.0"].argmax()])
    # change the order of columns
    metrics_df = metrics_df[["dataset", "adversary"] + METRIC_NAMES]
    print(metrics_df.to_markdown())
