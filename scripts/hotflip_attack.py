import argparse
from tqdm import tqdm
from pathlib import Path
import jsonlines
from datetime import datetime

import numpy as np
from allennlp.predictors import Predictor
from allennlp.common.util import dump_metrics

from adat.utils import load_jsonlines, calculate_wer
from adat.attackers import HotFlipFixed, AttackerOutput

parser = argparse.ArgumentParser()
parser.add_argument("--classifier-dir", type=str, required=True)
parser.add_argument("--test-path", type=str, required=True)
parser.add_argument("--out-dir", type=str, required=True)

parser.add_argument("--max-tokens", type=int, default=None)

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

    dump_metrics(args_path, args.__dict__)

    data = load_jsonlines(args.test_path)[:args.sample_size]
    predictor = Predictor.from_path(
        Path(args.classifier_dir) / "model.tar.gz",
        predictor_name="text_classifier",
        cuda_device=args.cuda
    )
    preds = predictor.predict_batch_json([{"sentence": el["text"].strip()} for el in data])

    attacker = HotFlipFixed(
        predictor=predictor,
        max_tokens=args.max_tokens or predictor._model.vocab.get_vocab_size("tokens")
    )

    print(f"Saving results to {results_path}")
    with jsonlines.open(results_path, "w") as writer:
        for el, p in tqdm(zip(data, preds)):

            # if it works then it's not stupid
            attacked_label = int(el["label"])
            probs = np.ones(predictor._model._num_labels)
            probs[attacked_label] = 0
            out = attacker.attack_from_json({"sentence": el["text"].strip()}, target={"probs": probs})
            adversarial_sequence = " ".join(out["final"][0])
            adversarial_probability = out["outputs"]["probs"]
            if len(adversarial_probability) == 1 and isinstance(adversarial_probability[0], list):
                adversarial_probabilities = adversarial_probability[0]
            else:
                adversarial_probabilities = adversarial_probability

            adversarial_probability = adversarial_probabilities[attacked_label]
            adversarial_label = int(np.argmax(adversarial_probabilities))

            adversarial_output = AttackerOutput(
                sequence=el["text"],
                probability=p["probs"][attacked_label],
                adversarial_sequence=adversarial_sequence,
                adversarial_probability=adversarial_probability,
                wer=calculate_wer(el["text"], adversarial_sequence),
                prob_diff=(p["probs"][attacked_label] - adversarial_probability),
                attacked_label=attacked_label,
                adversarial_label=adversarial_label
            )

            writer.write(adversarial_output.__dict__)
