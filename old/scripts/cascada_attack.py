import argparse
from datetime import datetime
import json
import jsonlines
from tqdm import tqdm
from pathlib import Path

from allennlp.common.util import dump_metrics

from adat.utils import load_jsonlines
from adat.attackers import Cascada, DistributionCascada

parser = argparse.ArgumentParser()
parser.add_argument("--config-path", type=str, required=True)
parser.add_argument("--lm-dir", type=str, required=True)
parser.add_argument("--classifier-dir", type=str, required=True)
parser.add_argument("--deep-levenshtein-dir", type=str, required=True)

parser.add_argument("--test-path", type=str, required=True)
parser.add_argument("--out-dir", type=str, required=True)

parser.add_argument("--sample-size", type=int, default=None)
parser.add_argument("--not-date-dir", action="store_true")
parser.add_argument("--force", action="store_true")
parser.add_argument("--distribution-level", action="store_true")
parser.add_argument("--cuda", type=int, default=-1)


if __name__ == "__main__":
    args = parser.parse_args()
    config = json.load(open(args.config_path))

    out_dir = Path(args.out_dir)
    if not args.not_date_dir:
        out_dir = out_dir / datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir.mkdir(exist_ok=True, parents=True)
    results_path = out_dir / "attacked_data.json"
    args_path = out_dir / "args.json"

    if not args.force:
        assert not results_path.exists()
        assert not args_path.exists()

    dump_metrics(str(args_path), {**args.__dict__, **config})

    data = load_jsonlines(args.test_path)[:args.sample_size]

    if args.distribution_level:
        cascada = DistributionCascada
    else:
        cascada = Cascada

    attacker = cascada(
        masked_lm_dir=args.lm_dir,
        classifier_dir=args.classifier_dir,
        deep_levenshtein_dir=args.deep_levenshtein_dir,
        alpha=config["alpha"],
        beta=config["beta"],
        lr=config["lr"],
        num_gumbel_samples=config.get("num_gumbel_samples", 1),
        tau=config.get("tau", 1.0),
        num_samples=config["num_samples"],
        temperature=config["temperature"],
        parameters_to_update=config["parameters_to_update"],
        device=args.cuda
    )

    print(f"Saving results to {results_path}")
    with jsonlines.open(results_path, "w") as writer:
        for el in tqdm(data):
            adversarial_output = attacker.attack(
                sequence_to_attack=el["text"],
                label_to_attack=el["label"],
                max_steps=config["max_steps"],
                early_stopping=config["early_stopping"]
            )

            writer.write(adversarial_output.__dict__)
