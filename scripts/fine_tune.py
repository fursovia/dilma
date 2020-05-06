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
            "--include-package",
            "adat",
            "--overrides",
            f'{"trainer.num_epochs": 1, "trainer.patience": 1, "train_data_path": {args.train_path}, "validation_data_path": {args.valid_path}, "distributed": {None}, "trainer.cuda_device": {args.cuda}}',
            "--recover"
        ]
    )
