import argparse
from tqdm import tqdm
import jsonlines
from pathlib import Path
from datetime import datetime

from allennlp.common.util import dump_metrics

from adat.utils import load_jsonlines
from adat.attackers import MaskedCascada

parser = argparse.ArgumentParser()
parser.add_argument("--lm-dir", type=str, required=True)
parser.add_argument("--classifier-dir", type=str, required=True)
parser.add_argument("--deep-levenshtein-dir", type=str, required=True)
parser.add_argument("--test-path", type=str, required=True)
parser.add_argument("--out-dir", type=str, required=True)

parser.add_argument("--max-steps", type=int, default=5)
parser.add_argument("--early-stopping", action="store_true", default=False)
parser.add_argument("--alpha", type=float, default=2.0)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--num-gumbel-samples", type=int, default=3)
parser.add_argument("--parameters-to-update", action="append", default=[])

parser.add_argument("--sample-size", type=int, default=None)
parser.add_argument("--not-date-dir", action="store_true")
parser.add_argument("--cuda", type=int, default=-1)


if __name__ == "__main__":
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not args.not_date_dir:
        out_dir = out_dir / datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir.mkdir(exist_ok=True, parents=True)
    results_path = out_dir / "attacked_data.json"
    args_path = out_dir / "args.json"
    assert not results_path.exists()
    assert not args_path.exists()

    dump_metrics(str(args_path), args.__dict__)

    data = load_jsonlines(args.test_path)[:args.sample_size]
    attacker = MaskedCascada(
        masked_lm_dir=args.lm_dir,
        classifier_dir=args.classifier_dir,
        deep_levenshtein_dir=args.deep_levenshtein_dir,
        alpha=args.alpha,
        lr=args.lr,
        num_gumbel_samples=args.num_gumbel_samples,
        parameters_to_update=args.parameters_to_update,
        device=args.cuda
    )

    with jsonlines.open(results_path, "w") as writer:
        for el in tqdm(data):
            adversarial_output = attacker.attack(
                sequence_to_attack=el["text"],
                label_to_attack=el["label"],
                max_steps=args.max_steps,
                early_stopping=args.early_stopping
            )

            writer.write(adversarial_output.__dict__)
