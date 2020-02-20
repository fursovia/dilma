import argparse
import json
import joblib
import pandas as pd
import numpy as np

from adat.utils import calculate_wer
from adat.models.classification_model import LogisticRegressionOnTfIdf
from sklearn.metrics import roc_auc_score

from adat.models import Task, get_model_by_name
from allennlp.data.vocabulary import Vocabulary
from adat.utils import load_weights
from adat.utils import calculate_perplexity
from adat.dataset import (
    ClassificationReader,
    CopyNetReader,
    LevenshteinReader,
    LanguageModelingReader,
    END_SYMBOL,
    START_SYMBOL
)

from adat.models import get_model_by_name
from adat.utils import load_weights, get_args_from_path
import os
from allennlp.data.vocabulary import Vocabulary
from adat.dataset import ClassificationReader, CopyNetReader, IDENTITY_TOKEN
from adat.predictors.classifier import ClassifierPredictor

parser = argparse.ArgumentParser()

parser.add_argument('-mp', '--model_path', type=str, help='path to text classifier')
parser.add_argument('-arp', '--attack_results_path', type=str, help='path adversarial attack results csv file')
parser.add_argument('-erp', '--eval_results_path', type=str, help='path to json file with results of evaluation')
parser.add_argument('-lmp', '--language_model_path', type=str, help='path to folder with trained language model')




def calculate_nad(labels, probs_orig, probs_gen, seqs_orig, seqs_gen):
    correct_inds = np.where(probs_orig.argmax(axis=1) == labels)[0]
    wers = np.array([calculate_wer(seqs_orig[i], seqs_gen[i]) for i in range(len(seqs_orig))])
    errs = (probs_orig.argmax(axis=1) != probs_gen.argmax(axis=1))
    return np.mean(errs[correct_inds]/(1e-6 + wers[correct_inds]))

def calculate_new_nad(labels, probs_orig, probs_gen, seqs_orig, seqs_gen, log_ppl_orig, log_ppl_adv, mode='normal'):
    correct_inds = np.where(probs_orig.argmax(axis=1) == labels)[0]
    wers = np.array([calculate_wer(seqs_orig[i], seqs_gen[i]) for i in range(len(seqs_orig))])
    errs = (probs_orig.argmax(axis=1) != probs_gen.argmax(axis=1))

    lens_orig = np.array([len(x.split()) for x in seqs_orig])
    lens_gen = np.array([len(x.split()) for x in seqs_gen])
    lens_max = np.maximum(lens_orig, lens_gen)

    if mode == 'normal':
        return np.mean(errs[correct_inds]*((lens_max[correct_inds] - wers[correct_inds])/(1e-6 + lens_max[correct_inds] - 1)))
    elif mode == 'divide_by_log_ppl_adv':
        return np.mean(errs[correct_inds]*((lens_max[correct_inds] - wers[correct_inds])/(1e-6 + lens_max[correct_inds] - 1))/log_ppl_adv[correct_inds])
    elif mode == 'multiply_by_log_ppl_ratio':
        log_ppl_ratio = log_ppl_orig[correct_inds] / log_ppl_adv[correct_inds]
        return np.mean(errs[correct_inds]*((lens_max[correct_inds] - wers[correct_inds])/(1e-6 + lens_max[correct_inds] - 1))*log_ppl_ratio)

def calculate_metrics(model, lm_model, reader, w, labels, seqs_orig, seqs_gen):
    probs_orig = model.predict(seqs_orig)
    probs_gen = model.predict(seqs_gen)
    acc_orig = (labels == (probs_orig*w).argmax(axis=1)).mean()
    acc_gen = (labels == (probs_gen*w).argmax(axis=1)).mean()
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
    metrics['NAD'] = calculate_nad(labels, probs_orig*w, probs_gen*w, seqs_orig, seqs_gen)

    log_ppl_orig = np.log(np.array(calculate_perplexity(seqs_orig, lm_model, reader)))
    log_ppl_adv = np.log(np.array(calculate_perplexity(seqs_gen, lm_model, reader)))
    metrics['NAD_new'] = calculate_new_nad(labels, probs_orig*w, probs_gen*w, seqs_orig, seqs_gen, log_ppl_orig, log_ppl_adv)
    metrics['NAD_new/log_ppl_adv'] = calculate_new_nad(labels, probs_orig*w, probs_gen*w, seqs_orig, seqs_gen, log_ppl_orig, log_ppl_adv, mode='divide_by_log_ppl_adv')
    metrics['NAD_new*log_ppl_ratio'] = calculate_new_nad(labels, probs_orig*w, probs_gen*w, seqs_orig, seqs_gen, log_ppl_orig, log_ppl_adv, mode='multiply_by_log_ppl_ratio')
    metrics['log_ppl_adv'] = np.mean(log_ppl_adv)
    metrics['log_ppl_orig'] = np.mean(log_ppl_orig)
    return metrics

if __name__ == '__main__':
    args = parser.parse_args()
    
    model_with_weights = joblib.load(args.model_path)
    model = model_with_weights['model']
    w = model_with_weights['weights']
    
    df = pd.read_csv(args.attack_results_path)
    df.rename(columns={'generated_sequence':'adversarial_sequence'}, inplace=True)


    reader = LanguageModelingReader()
    lm_path = os.path.join(args.language_model_path, 'best.th')
    lm_vocab_path = os.path.join(args.language_model_path, 'vocab')
    vocab = Vocabulary.from_files(lm_vocab_path)
    lm_model = get_model_by_name('lm', vocab=vocab)
    load_weights(lm_model, lm_path)
    
    metrics = calculate_metrics(model, lm_model, reader,
                                w,
                                df['label'].values, 
                                df['sequence'].values,
                                df['adversarial_sequence'].values)
    
    print(metrics)
    json.dump(metrics, open(args.eval_results_path, 'w'))
