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
    metrics['accuracy_drop'] = acc_orig - acc_gen
    metrics['probability_drop'] = (proba_orig - proba_gen).mean() 
    metrics['WER'] = np.mean([calculate_wer(seqs_orig[i], seqs_gen[i]) 
                              for i in range(len(seqs_orig))])
    metrics['NAD'] = calculate_nad(labels, probs_orig, probs_gen, seqs_orig, seqs_gen)
    return metrics

if __name__ == '__main__':
    args = parser.parse_args()
    
    model = joblib.load(args.model_path)
    
    df = pd.read_csv(args.attack_results_path)
    
    metrics = calculate_metrics(model, 
                                df['labels'].values, 
                                df['sequences'].values,
                                df['generated_sequence'].values)
    
    
    json.dump(metrics, open(args.eval_results_path, 'w'))
