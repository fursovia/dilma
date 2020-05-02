import argparse
from pathlib import Path
from tqdm import tqdm
import jsonlines

import numpy as np
from sklearn.model_selection import train_test_split

from adat.utils import calculate_normalized_wer, load_jsonlines, SequenceModifier

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--field-name", type=str, default="text")
parser.add_argument("--num-adversarial", type=int, default=200000)
parser.add_argument("--num-non-adversarial", type=int, default=30000)
parser.add_argument("--test-size", type=float, default=0.15)


if __name__ == "__main__":
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    train_path = output_dir / "train.json"
    test_path = output_dir / "test.json"
    assert not train_path.exists() and not test_path.exists()

    data = load_jsonlines(args.data_path)
    sequences = [str(el[args.field_name]) for el in data]
    vocab = []
    for seq in sequences:
        vocab.extend(seq.split())
    vocab = list(set(vocab))
    modifier = SequenceModifier(vocab)

    dataset = []
    non_adversarial_indexes = np.random.randint(0, len(sequences), size=(args.num_non_adversarial, 2))
    for id1, id2 in tqdm(non_adversarial_indexes):
        tr1 = sequences[id1]
        tr2 = sequences[id2]
        dist = calculate_normalized_wer(tr1, tr2)
        ex = {"seq_a": tr1, "seq_b": tr2, "dist": dist}
        dataset.append(ex)

    adversarial_indexes = np.random.randint(0, len(sequences), size=(args.num_adversarial, ))
    for idx in tqdm(adversarial_indexes):
        tr1 = sequences[idx]
        tr2 = modifier(tr1)
        dist = calculate_normalized_wer(tr1, tr2)
        ex = {"seq_a": tr1, "seq_b": tr2, "dist": dist}
        dataset.append(ex)

    train, test = train_test_split(dataset, test_size=args.test_size)

    with jsonlines.open(train_path, "w") as writer:
        for ex in train:
            writer.write(ex)

    with jsonlines.open(test_path, "w") as writer:
        for ex in test:
            writer.write(ex)
