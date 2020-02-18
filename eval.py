import argparse
import json
import joblib
import pandas as pd
import numpy as np

from adat.utils import calculate_wer
from adat.models.classification_model import LogisticRegressionOnTfIdf

parser = argparse.ArgumentParser()

parser.add_argument('-mp', '--model_path', type=str, help='path to text classifier')
parser.add_argument('-arp', '--attack_results_path', type=str, help='path adversarial attack results csv file')
parser.add_argument('-erp', '--eval_results_path', type=str, help='path to json file with results of evaluation')

def calculate_nad(labels, probs_orig, probs_gen, seqs_orig, seqs_gen):
    correct_inds = np.where(probs_orig.argmax(axis=1) == labels)[0]
    wers = np.array([calculate_wer(seqs_orig[i], seqs_gen[i]) for i in range(len(seqs_orig))])
    errs = (probs_orig.argmax(axis=1) != probs_gen.argmax(axis=1))
    return np.mean(errs[correct_inds]/(1e-6 + wers[correct_inds]))

def calculate_metrics(model, labels, seqs_orig, seqs_gen):
    probs_orig = model.predict(seqs_orig)
    probs_gen = model.predict(seqs_gen)
    acc_orig = (labels == probs_orig.argmax(axis=1)).mean()
    acc_gen = (labels == probs_gen.argmax(axis=1)).mean()
    proba_orig = probs_orig[np.arange(len(labels)), labels]
    proba_gen = probs_gen[np.arange(len(labels)), labels]
    metrics = {}
    if len(set(test_y)) > 2:
        auc_orig = roc_auc_score(y_true=test_y, y_score=proba_orig, multi_class='ovr', average='macro')
        auc_gen = roc_auc_score(y_true=test_y, y_score=proba_gen, multi_class='ovr', average='macro')
    else:
        auc_orig = roc_auc_score(y_true=test_y, y_score=proba_orig[:, 1])
        auc_gen = roc_auc_score(y_true=test_y, y_score=auc_gen[:, 1])
    metrics['accuracy_drop'] = acc_orig - acc_gen
    metrics['roc_auc_drop'] = auc_orig - auc_gen
    metrics['probability_drop'] = (proba_orig - proba_gen).mean() 
    metrics['WER'] = np.mean([calculate_wer(seqs_orig[i], seqs_gen[i]) 
                              for i in range(len(seqs_orig))])
    metrics['NAD'] = calculate_nad(labels, probs_orig, probs_gen, seqs_orig, seqs_gen)
    return metrics

if __name__ == '__main__':
    args = parser.parse_args()
    
    model = joblib.load(args.model_path)
    
    df = pd.read_csv(args.attack_results_path)
    df.rename(columns={'generated_sequence':'adversarial_sequence'}, inplace=True)
    
    metrics = calculate_metrics(model, 
                                df['label'].values, 
                                df['sequence'].values,
                                df['adversarial_sequence'].values)
    
    print(metrics)
    json.dump(metrics, open(args.eval_results_path, 'w'))
