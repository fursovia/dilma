import argparse
from pathlib import Path
import jsonlines

from adat.utils import load_jsonlines
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--adversarial-dir", type=str, required=True)
parser.add_argument("--out-dir", type=str, required=True)


if __name__ == "__main__":
    args = parser.parse_args()
    adversarial_dir = Path(args.adversarial_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    data = load_jsonlines(adversarial_dir / "attacked_data.json")
    train, test = train_test_split(data, test_size=0.2, random_state=24)

    print(f"Saving data to {out_dir}")
    with jsonlines.open(out_dir / "train.json", "w") as writer:
        for ex in train:
            if ex["wer"] > 0 and ex["attacked_label"] != ex["adversarial_label"]:
                writer.write({"text": ex["adversarial_sequence"], "label": 1})
                writer.write({"text": ex["sequence"], "label": 0})

    with jsonlines.open(out_dir / "test.json", "w") as writer:
        for ex in test:
            if ex["wer"] > 0 and ex["attacked_label"] != ex["adversarial_label"]:
                writer.write({"text": ex["adversarial_sequence"], "label": 1})
                writer.write({"text": ex["sequence"], "label": 0})
