import argparse
from pathlib import Path
from tqdm import tqdm
import random
import jsonlines

import numpy as np
from sklearn.model_selection import train_test_split

from adat.utils import calculate_wer, load_jsonlines, SequenceModifier

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--field-name", type=str, default="text")
parser.add_argument("--num-adversarial", type=int, default=200000)
parser.add_argument("--num-non-adversarial", type=int, default=30000)
parser.add_argument("--add-prob", type=float, default=None)
parser.add_argument("--replace-prob", type=float, default=None)
parser.add_argument("--remove-prob", type=float, default=None)
parser.add_argument("--test-size", type=float, default=0.15)

REMOVE_PROB = 0.0
ADD_PROB = 0.0
REPLACE_PROB = 1.0
NUM_SMALL_CHANGES = 3
NUM_SEQUENTIAL_CHANGES = 10

if __name__ == "__main__":
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    train_path = output_dir / "train.json"
    test_path = output_dir / "test.json"
    assert not train_path.exists() and not test_path.exists()

    data_dir = Path(args.data_dir)
    data = load_jsonlines(data_dir / "train.json") + load_jsonlines(data_dir / "test.json")
    sequences = [str(el[args.field_name]) for el in data]
    mean_len = float(np.mean([len(seq.split()) for seq in sequences]))
    vocab = []
    for seq in sequences:
        vocab.extend(seq.split())
    vocab = list(set(vocab))

    modifier = SequenceModifier(
        vocab,
        remove_prob=args.remove_prob or REMOVE_PROB,
        add_prob=args.add_prob or ADD_PROB,
        replace_prob=args.replace_prob or REPLACE_PROB
    )

    dataset = []
    non_adversarial_indexes = np.random.randint(0, len(sequences), size=(args.num_non_adversarial, 2))
    for id1, id2 in tqdm(non_adversarial_indexes):
        tr1 = sequences[id1].strip()
        tr2 = sequences[id2].strip()
        dist = calculate_wer(tr1, tr2)
        ex = {"seq_a": tr1, "seq_b": tr2, "dist": dist}
        dataset.append(ex)

    adversarial_indexes = np.random.randint(0, len(sequences), size=(args.num_adversarial, ))
    for idx in tqdm(adversarial_indexes):
        tr1 = sequences[idx].strip()

        for _ in range(NUM_SMALL_CHANGES):
            tr2 = modifier(tr1).strip()
            dist = calculate_wer(tr1, tr2)
            ex = {"seq_a": tr1, "seq_b": tr2, "dist": dist}
            dataset.append(ex)

        tr2 = tr1
        num_changes = NUM_SEQUENTIAL_CHANGES
        if num_changes > len(tr1.split()):
            num_changes = len(tr1.split())
        for _ in range(num_changes):
            position_idx = random.randint(0, len(tr1.split()) - 1)
            replace_with = random.choice(vocab)
            tr2 = tr2.split()
            tr2[position_idx] = replace_with
            tr2 = " ".join(tr2)

            dist = calculate_wer(tr1, tr2)
            ex = {"seq_a": tr1, "seq_b": tr2, "dist": dist}
            dataset.append(ex)

    train, test = train_test_split(dataset, test_size=args.test_size)
    with jsonlines.open(train_path, "w") as writer:
        for ex in train:
            writer.write(ex)

    with jsonlines.open(test_path, "w") as writer:
        for ex in test:
            writer.write(ex)
