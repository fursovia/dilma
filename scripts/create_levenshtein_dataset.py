import argparse
from pathlib import Path
import jsonlines

import numpy as np
import pandas as pd
from adat.utils import calculate_normalized_wer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--csv_path", type=str, required=True)
parser.add_argument("--col_name", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--train_size", type=int, default=200000)
parser.add_argument("--test_size", type=int, default=10000)


if __name__ == "__main__":
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    train_path = output_dir / "train.json"
    test_path = output_dir / "test.json"
    assert not train_path.exists() and not test_path.exists()

    data = pd.read_csv(args.csv_path)
    data = data[~data[args.col_name].isna()]
    sequences = data[args.col_name].astype(str).tolist()

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
