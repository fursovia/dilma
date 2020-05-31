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
    for dataset in metrics_df["dataset"].unique():
        df = metrics_df[metrics_df["dataset"] == dataset]
        best_NAD = df["NAD_1.0"].max()
        best_config_num = df.iloc[df["NAD_1.0"].argmax()]["config_num"]

        print(f"Dataset: {dataset}, Best NAD: {best_NAD}, Best config num: {best_config_num}")
