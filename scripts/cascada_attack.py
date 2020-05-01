import argparse
from tqdm import tqdm
import jsonlines

from adat.utils import load_jsonlines
from adat.attackers import MaskedCascada

parser = argparse.ArgumentParser()
parser.add_argument("--lm-dir", type=str, required=True)
parser.add_argument("--classifier-dir", type=str, required=True)
parser.add_argument("--deep-leveneshtein-dir", type=str, required=True)
parser.add_argument("--test-path", type=str, required=True)
parser.add_argument("--out-path", type=str, required=True)

parser.add_argument("--max-steps", type=int, default=10)
parser.add_argument("--thresh-drop", type=float, default=0.2)
parser.add_argument("--early-stopping", action="store_true", default=False)
parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--cuda", type=int, default=-1)


if __name__ == "__main__":
    args = parser.parse_args()
    data = load_jsonlines(args.test_path)

    attacker = MaskedCascada(
        masked_lm_dir=args.lm_dir,
        classifier_dir=args.classifier_dir,
        deep_levenshtein_dir=args.deep_levenshtein_dir,
        alpha=args.alpha,
        lr=args.lr,
        device=args.cuda
    )

    with jsonlines.open(args.out_path) as writer:
        for el in tqdm(data):
            adversarial_output = attacker.attack(
                sequence_to_attack=el["text"],
                label_to_attack=el["label"],
                max_steps=args.max_steps,
                thresh_drop=args.thresh_drop,
                early_stopping=args.early_stopping
            )

            writer.write(adversarial_output.__dict__)
