import os
import argparse
from pathlib import Path
import json
import shutil


parser = argparse.ArgumentParser()
parser.add_argument("--log-dir", type=str, required=True)
parser.add_argument("--fine-tune-dir", type=str, required=True)
parser.add_argument("--train-path", type=str, required=True)
parser.add_argument("--cuda", type=int, default=-1)


if __name__ == "__main__":
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    fine_tune_dir = Path(args.fine_tune_dir)
    fine_tune_dir.mkdir(exist_ok=True)

    config = json.load(open(log_dir / "config.json"))
    config["train_data_path"] = args.train_path
    config.pop("distributed")
    config["trainer"]["num_epochs"] = 1
    config["trainer"]["patience"] = 1
    config["trainer"]["cuda_device"] = args.cuda

    with open(fine_tune_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    shutil.copytree(str(log_dir / "vocabulary"), str(fine_tune_dir / "vocabulary"))
    shutil.copy(str(log_dir / "best.th"), str(fine_tune_dir / "best.th"))

    os.system(
        " ".join(
            [
                "allennlp",
                "train",
                str(fine_tune_dir / "config.json"),
                "--serialization-dir",
                f"{fine_tune_dir}",
                "--include-package",
                "adat",
                "--recover"
            ]
        )
    )
