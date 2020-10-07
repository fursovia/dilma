import argparse
from tqdm import tqdm
from pathlib import Path
import jsonlines
import json
from datetime import datetime

from allennlp.common.util import dump_metrics

from adat.utils import load_jsonlines
from adat.attackers import FGSMAttacker, DeepFoolAttacker

parser = argparse.ArgumentParser()
parser.add_argument("--config-path", type=str, required=True)
parser.add_argument("--classifier-dir", type=str, required=True)
parser.add_argument("--test-path", type=str, required=True)
parser.add_argument("--out-dir", type=str, required=True)

parser.add_argument("--attacker", type=str, choices=["fgsm", "deepfool"], required=True)

parser.add_argument("--sample-size", type=int, default=None)
parser.add_argument("--not-date-dir", action="store_true")
parser.add_argument("--force", action="store_true")
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

    if args.attacker == "fgsm":
        attacker = FGSMAttacker(args.classifier_dir, device=args.cuda, **config)
    elif args.attacker == "deepfool":
        attacker = DeepFoolAttacker(args.classifier_dir, device=args.cuda, **config)
    else:
        raise NotImplementedError

    print(f"Saving results to {results_path}")
    with jsonlines.open(results_path, "w") as writer:
        for el in tqdm(data):
            adversarial_output = attacker.attack(
                sequence_to_attack=el["text"],
                label_to_attack=el["label"]
            )

            writer.write(adversarial_output.__dict__)
