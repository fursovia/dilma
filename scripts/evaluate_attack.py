import argparse
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
import json

import numpy as np
from allennlp.predictors import Predictor
from allennlp.models import Model
from allennlp.data import DatasetReader
from allennlp.common.params import Params

from adat.dataset_readers.lm_reader import SimpleLanguageModelingDatasetReaderFixed
from adat.utils import load_jsonlines, normalized_accuracy_drop, normalized_accuracy_drop_with_perplexity

parser = argparse.ArgumentParser()
parser.add_argument("--adversarial-dir", type=str, required=True)
parser.add_argument("--classifier-dir", type=str, required=True)
parser.add_argument("--lm-dir", type=str, default=None)
parser.add_argument("--gamma", type=float, default=1.0)
parser.add_argument("--cuda", type=int, default=-1)


if __name__ == "__main__":
    args = parser.parse_args()
    adversarial_dir = Path(args.adversarial_dir)
    data = load_jsonlines(adversarial_dir / "attacked_data.json")

    if args.lm_dir is not None:
        lm_dir = Path(args.lm_dir)
        lm_model = Model.from_archive(lm_dir / "model.tar.gz")
        reader = DatasetReader.from_params(Params.from_file(lm_dir /'config.json')['dataset_reader'])
        lm_predictor = Predictor(lm_model, reader)
        # lm_predictor = Predictor.from_path(
        #     lm_dir / "model.tar.gz",
        #     # this is not a mistake
        #     predictor_name="text_classifier",
        #     cuda_device=args.cuda
        # )
        lm_predictor._model._tokens_masker = None
        get_perplexity = lambda text: np.exp(
            lm_predictor.predict_instance(
                lm_predictor._dataset_reader.text_to_instance(text)
            )["loss"]
        )
        orig_perplexities = [get_perplexity(el["sequence"]) for el in tqdm(data)]
        adv_perplexities = [get_perplexity(el["adversarial_sequence"]) for el in tqdm(data)]
        perp_diff = [max(0.0, adv_perplexities[i] - orig_perplexities[i]) for i in range(len(data))]
        mean_perplexity_rise = float(np.mean(perp_diff))
    else:
        mean_perplexity_rise = None
        orig_perplexities = None
        adv_perplexities = None

    classifier_dir = Path(args.classifier_dir)
    predictor = Predictor.from_path(
        classifier_dir / "model.tar.gz",
        predictor_name="text_classifier",
        cuda_device=args.cuda
    )
    preds = predictor.predict_batch_json([{"sentence": el["sequence"]} for el in data])
    y_true = [int(el["attacked_label"]) for el in data]
    correctly_predicted = [y == int(np.argmax(p["probs"])) for y, p in zip(y_true, preds)]

    adv_preds = predictor.predict_batch_json([{"sentence": el["adversarial_sequence"]} for el in data])
    y_adv = [int(np.argmax(el["probs"])) for el in adv_preds]

    prob_diffs = [p["probs"][l] - ap["probs"][l] for p, ap, l in zip(preds, adv_preds, y_true)]
    wers = [el["wer"] for el in data]

    nad = normalized_accuracy_drop(
        wers=wers,
        y_true=y_true,
        y_adv=y_adv,
        gamma=args.gamma
    )

    if orig_perplexities is not None and adv_perplexities is not None:
        nad_with_perp = normalized_accuracy_drop_with_perplexity(
            wers=wers,
            y_true=y_true,
            y_adv=y_adv,
            perp_true=orig_perplexities,
            perp_adv=adv_perplexities,
            gamma=args.gamma
        )
    else:
        nad_with_perp = None

    metrics = dict(
        mean_prob_diff=float(np.mean(prob_diffs)),
        mean_wer=float(np.mean(wers)),
        mean_perplexity_rise=mean_perplexity_rise
    )
    metrics[f"NAD_{args.gamma}"] = nad
    metrics[f"NAD_with_perplexity_{args.gamma}"] = nad_with_perp
    metrics["path_to_classifier"] = str(classifier_dir.absolute())
    if args.lm_dir is not None:
        metrics["path_to_lm"] = str(Path(args.lm_dir).absolute())
    else:
        metrics["path_to_lm"] = None

    pprint(metrics)
    with open(adversarial_dir / f"{classifier_dir.name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
