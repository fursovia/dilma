import argparse
from pathlib import Path
import jsonlines

from adat.utils import load_jsonlines

parser = argparse.ArgumentParser()
parser.add_argument("--adversarial-dir", type=str, required=True)
parser.add_argument("--mix-with-path", type=str, default=None)
parser.add_argument("--max-wer", type=int, default=3)


if __name__ == "__main__":
    args = parser.parse_args()
    adversarial_dir = Path(args.adversarial_dir)

    data = load_jsonlines(adversarial_dir / "attacked_data.json")

    num_added = 0
    data_path = adversarial_dir / "fine_tuning_data.json"
    print(f"Saving data to {data_path}")
    with jsonlines.open(data_path, "w") as writer:
        for ex in data:
            if ex["wer"] <= args.max_wer and ex["attacked_label"] != ex["adversarial_label"]:
                num_added += 1
                writer.write({"text": ex["adversarial_sequence"], "label": ex["attacked_label"]})

        if args.mix_with_path is not None:
            data_to_mix_with = load_jsonlines(args.min_with_path)
            for ex in data_to_mix_with:
                writer.write(ex)

    print(f"Saved {num_added} examples.")
