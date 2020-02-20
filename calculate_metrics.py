import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

from adat.utils import calculate_wer

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, help='path to text classifier')
parser.add_argument('--results_dir', type=str, help='path adversarial attack results csv file')


def calculate_nad(labels, probs_orig, probs_gen, seqs_orig, seqs_gen):
    correct_inds = np.where(probs_orig.argmax(axis=1) == labels)[0]
    wers = np.array([calculate_wer(seqs_orig[i], seqs_gen[i]) for i in range(len(seqs_orig))])
    errs = (probs_orig.argmax(axis=1) != probs_gen.argmax(axis=1))
    return np.mean(errs[correct_inds] / (1e-6 + wers[correct_inds]))


def calculate_new_nad(labels, probs_orig, probs_gen, seqs_orig, seqs_gen):
    correct_inds = np.where(probs_orig.argmax(axis=1) == labels)[0]
    wers = np.array([calculate_wer(seqs_orig[i], seqs_gen[i]) for i in range(len(seqs_orig))])
    errs = (probs_orig.argmax(axis=1) != probs_gen.argmax(axis=1))

    lens_orig = np.array([len(x.split()) for x in seqs_orig])
    lens_gen = np.array([len(x.split()) for x in seqs_gen])
    lens_max = np.maximum(lens_orig, lens_gen)
    return np.mean(
        errs[correct_inds] * ((lens_max[correct_inds] - wers[correct_inds]) / (1e-6 + lens_max[correct_inds] - 1)))


def calculate_metrics(model, w, labels, seqs_orig, seqs_gen):
    probs_orig = model.predict(seqs_orig)
    probs_gen = model.predict(seqs_gen)
    acc_orig = (labels == (probs_orig * w).argmax(axis=1)).mean()
    acc_gen = (labels == (probs_gen * w).argmax(axis=1)).mean()
    proba_orig = probs_orig[np.arange(len(labels)), labels]
    proba_gen = probs_gen[np.arange(len(labels)), labels]
    metrics = {}
    if len(set(labels)) > 2:
        auc_orig = roc_auc_score(y_true=labels, y_score=probs_orig, multi_class='ovr', average='macro')
        auc_gen = roc_auc_score(y_true=labels, y_score=probs_gen, multi_class='ovr', average='macro')
    else:
        auc_orig = roc_auc_score(y_true=labels, y_score=probs_orig[:, 1])
        auc_gen = roc_auc_score(y_true=labels, y_score=probs_gen[:, 1])
    metrics['accuracy_drop'] = acc_orig - acc_gen
    metrics['roc_auc_drop'] = auc_orig - auc_gen
    metrics['probability_drop'] = (proba_orig - proba_gen).mean()
    metrics['WER'] = np.mean([calculate_wer(seqs_orig[i], seqs_gen[i])
                              for i in range(len(seqs_orig))])
    metrics['NAD'] = calculate_nad(labels, probs_orig * w, probs_gen * w, seqs_orig, seqs_gen)
    metrics['NAD_new'] = calculate_new_nad(labels, probs_orig * w, probs_gen * w, seqs_orig, seqs_gen)
    return metrics


if __name__ == '__main__':
    args = parser.parse_args()

    model_with_weights = joblib.load(args.model_path)
    model = model_with_weights['model']
    w = model_with_weights['weights']

    df = pd.read_csv(Path(args.results_dir) / 'results.csv')
    df.rename(columns={'generated_sequence': 'adversarial_sequence'}, inplace=True)

    metrics = calculate_metrics(model,
                                w,
                                df['label'].values,
                                df['sequence'].values,
                                df['adversarial_sequence'].values)

    print(metrics)
    json.dump(metrics, open(Path(args.results_dir) / 'final_metrics.json', 'w'))
