import argparse
from tqdm import tqdm
from pathlib import Path
import jsonlines

import numpy as np
from allennlp.predictors import Predictor

from adat.utils import load_jsonlines, calculate_wer
from adat.attackers import HotFlipFixed, AttackerOutput

parser = argparse.ArgumentParser()
parser.add_argument("--classifier-dir", type=str, required=True)
parser.add_argument("--test-path", type=str, required=True)
parser.add_argument("--out-path", type=str, required=True)

parser.add_argument("--max-tokens", type=int, default=None)

parser.add_argument("--sample-size", type=int, default=None)
parser.add_argument("--cuda", type=int, default=-1)


if __name__ == "__main__":
    args = parser.parse_args()
    classifier_dir = Path(args.classifier_dir)
    data = load_jsonlines(args.test_path)[:args.sample_size]

    predictor = Predictor.from_path(
        classifier_dir / "model.tar.gz",
        predictor_name="text_classifier",
        cuda_device=args.cuda
    )
    preds = predictor.predict_batch_json([{"sentence": el["text"]} for el in data])

    attacker = HotFlipFixed(predictor=predictor, max_tokens=args.max_tokens)

    with jsonlines.open(args.out_path, "w") as writer:
        for el, p in tqdm(zip(data, preds)):

            # if it works then it's not stupid
            attacked_label = int(el["label"])
            probs = np.ones(predictor._model._num_labels)
            probs[attacked_label] = 0
            out = attacker.attack_from_json({"sentence": el["text"]}, target={"probs": probs})
            adversarial_sequence = " ".join(out["final"][0])
            adversarial_probability = out["outputs"]["probs"][int(el["label"])]
            adversarial_output = AttackerOutput(
                sequence=el["text"],
                probability=p["probs"][attacked_label],
                adversarial_sequence=adversarial_sequence,
                adversarial_probability=adversarial_probability,
                wer=calculate_wer(el["text"], adversarial_sequence),
                prob_diff=(p["probs"][attacked_label] - adversarial_probability),
                attacked_label=attacked_label
            )

            writer.write(adversarial_output.__dict__)
