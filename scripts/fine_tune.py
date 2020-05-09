import os
import argparse
from pathlib import Path
import json
import shutil

from allennlp.models.archival import archive_model

parser = argparse.ArgumentParser()
parser.add_argument("--log-dir", type=str, required=True)
parser.add_argument("--fine-tune-dir", type=str, required=True)
parser.add_argument("--train-path", type=str, required=True)
parser.add_argument("--cuda", type=int, default=-1)
parser.add_argument("--lr", type=float, default=0.000001)


if __name__ == "__main__":
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    fine_tune_dir = Path(args.fine_tune_dir)
    shutil.copytree(str(log_dir), str(fine_tune_dir))
    
    states = fine_tune_dir.glob("model_state_epoch_*.th")
    last_epoch = max([int(p.name.split("_")[-1].split(".")[0]) for p in states])
    
    config = json.load(open(log_dir / "config.json"))
    config.pop("validation_data_path")
    config["train_data_path"] = args.train_path
    config.pop("distributed")
    config["trainer"]["num_epochs"] = last_epoch + 2
    config["trainer"].pop("patience")
    config["trainer"]["cuda_device"] = args.cuda
    config["trainer"]["optimizer"] = {"type": "adam", "lr": args.lr}

    with open(fine_tune_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

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
    print(f"Saving `model_state_epoch_{last_epoch + 1}.th` to archive")
    archive_model(str(fine_tune_dir), f"model_state_epoch_{last_epoch + 1}.th")
