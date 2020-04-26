import argparse
from tqdm import tqdm
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from adat.utils import calculate_normalized_wer
from adat.masker import get_default_masker


parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--num_zero', type=int, default=200000)
parser.add_argument('--num_non_zero', type=int, default=1000000)
parser.add_argument('--test_size', type=float, default=0.05)


if __name__ == '__main__':
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    train_path = output_dir / 'train.csv'
    test_path = output_dir / 'test.csv'
    assert not train_path.exists() and not test_path.exists()

    data = pd.read_csv(args.csv_path)
    sequences = data['sequences'].values

    masker = get_default_masker()

    close_to_zero_examples = []
    num_close_to_zero = args.num_zero
    close_to_zero_indexes = np.random.randint(0, len(sequences), size=(num_close_to_zero, 2))
    for id1, id2 in tqdm(close_to_zero_indexes):
        tr1 = sequences[id1]
        tr2 = sequences[id2]
        wer_sim = 1 - calculate_normalized_wer(tr1, tr2)
        close_to_zero_examples.append((tr1, tr2, wer_sim))

    non_zero_examples = []
    num_non_zero_examples = args.num_non_zero
    non_zero_examples_indexes = np.random.randint(0, len(sequences), size=num_non_zero_examples)
    for idx in tqdm(non_zero_examples_indexes):
        tr1 = sequences[idx]
        tr2, applied = masker.mask(tr1)
        if applied:
            wer_sim = 1 - calculate_normalized_wer(tr1, tr2)
            non_zero_examples.append((tr1, tr2, wer_sim))

    examples = []
    examples.extend(close_to_zero_examples)
    examples.extend(non_zero_examples)
    examples = pd.DataFrame(examples, columns=['seq_a', 'seq_b', 'similarity'])
    examples = examples.sample(frac=1).reset_index(drop=True)

    train, test = train_test_split(examples, test_size=args.test_size, random_state=24)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
