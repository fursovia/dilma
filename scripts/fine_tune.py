import subprocess
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--log-dir", type=str, required=True)
parser.add_argument("--train-path", type=str, required=True)
parser.add_argument("--valid-path", type=str, required=True)
parser.add_argument("--cuda", type=int, default=-1)


if __name__ == "__main__":
    args = parser.parse_args()
    log_dir = Path(args.log_dir)

    subprocess.run(
        [
            "allennlp",
            "train",
            str(log_dir / "config.json"),
            "--serialization-dir",
            "{args.log_dir}"
            "--include-package adat",
            "--overrides",
            '{"trainer.num_epochs": 1, "trainer.patience": 1, \
            "train_data_path": TRAIN_PATH, \
            "validation_data_path": VALID_PATH, \
            "distributed": null, \
            "trainer.cuda_device": CUDA}'.replace(
                "TRAIN_PATH", args.train_path
            ).replace(
                "VALID_PATH", args.valid_path
            ).replace(
                "CUDA", str(args.cuda)
            ),
            "--recover"
        ]
    )
