import argparse
from pathlib import Path
import jsonlines

import numpy as np
from adat.utils import calculate_normalized_wer, load_jsonlines
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--field-name", type=str, default="text")
parser.add_argument("--train-size", type=int, default=200000)
parser.add_argument("--test-size", type=int, default=10000)


if __name__ == "__main__":
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    train_path = output_dir / "train.json"
    test_path = output_dir / "test.json"
    assert not train_path.exists() and not test_path.exists()

    data = load_jsonlines(args.data_path)
    sequences = [str(el[args.field_name]) for el in data]

    train_indexes = np.random.randint(0, len(sequences), size=(args.train_size, 2))
    with jsonlines.open(train_path, "w") as writer:
        for id1, id2 in tqdm(train_indexes):
            tr1 = sequences[id1]
            tr2 = sequences[id2]
            dist = calculate_normalized_wer(tr1, tr2)
            ex = {"seq_a": tr1, "seq_b": tr2, "dist": dist}
            writer.write(ex)

    test_indexes = np.random.randint(0, len(sequences), size=(args.test_size, 2))
    with jsonlines.open(test_path, "w") as writer:
        for id1, id2 in tqdm(test_indexes):
            tr1 = sequences[id1]
            tr2 = sequences[id2]
            dist = calculate_normalized_wer(tr1, tr2)
            ex = {"seq_a": tr1, "seq_b": tr2, "dist": dist}
            writer.write(ex)
