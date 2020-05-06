import os
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

    overrides = {
        "trainer.num_epochs": 1,
        "trainer.patience": 1,
        "train_data_path": args.train_path,
        "validation_data_path": args.valid_path,
        "distributed": "null",
        "trainer.cuda_device": args.cuda

    }

    os.system(
        " ".join(
            [
                "allennlp",
                "train",
                str(log_dir / "config.json"),
                "--serialization-dir",
                "{args.log_dir}",
                "--include-package",
                "adat",
                "--overrides",
                str(overrides),
                "--recover"
            ]
        )
    )
