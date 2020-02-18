import argparse
import json
import os
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, f1_score
from adat.models.classification_model import LogisticRegressionOnTfIdf

parser = argparse.ArgumentParser()

parser.add_argument('-dd', '--dataset_dir', type=str, help='path to dataset with train.csv and test.csv files')
parser.add_argument('-mp', '--model_path', type=str, help='path to store classifier')
parser.add_argument('-mqp', '--model_quality_path', type=str, default=None, help='path to store classifier\'s performance metric')
parser.add_argument('-mt', '--model_type', type=str, default='logit_tfidf', help='alias of the model type')


if __name__ == '__main__':
    args = parser.parse_args()
    
    if args.model_type == 'logit_tfidf':
        train_data = pd.read_csv(os.path.join(args.dataset_dir, 'train.csv'))
        test_data = pd.read_csv(os.path.join(args.dataset_dir, 'test.csv'))
        

        train_inds = np.random.choice(len(train_data), int(0.7*len(train_data)), replace=False)
        val_inds = np.setdiff1d(np.arange(len(train_data)), train_inds)

        train_x = train_data.sequences.values[train_inds]
        train_y = train_data.labels.values[train_inds]

        test_x = train_data.sequences.values[val_inds]
        test_y = train_data.labels.values[val_inds]

        model = LogisticRegressionOnTfIdf()
        model.fit(train_x, train_y)

        # tune proba weights

        probs = model.predict(test_x)

        f1s = []
        ws = []
        for _ in tqdm(range(2000)):
            w = np.random.rand(probs.shape[1])
            ws.append(w)
            if len(test_y) > 2:
                f1 = f1_score(test_y, (probs*w).argmax(axis=1), average='macro')
            else:
                f1 = f1_score(test_y, (probs*w).argmax(axis=1))
            f1s.append(f1)

        best_w = ws[np.argmax(f1s)]


        # fit full model
        train_x = train_data.sequences.values
        train_y = train_data.labels.values

        test_x = test_data.sequences.values
        test_y = test_data.labels.values

        model = LogisticRegressionOnTfIdf()
        model.fit(train_x, train_y)
        
        joblib.dump({'model':model, 'weights':best_w}, args.model_path)
        
        probs = model.predict(test_x)
        
        if len(set(test_y)) > 2:
            auc = roc_auc_score(y_true=test_y, y_score=probs, multi_class='ovr', average='macro')
        else:
            auc = roc_auc_score(y_true=test_y, y_score=probs[:, 1])
        acc = (test_y == (probs*best_w).argmax(axis=1)).mean()
        acc_random = (test_y == np.random.choice(test_y, len(test_y), replace=False)).mean()

        metric = {
            'ROC AUC':auc,
            'Accuracy':acc,
            'Accuracy(random)':acc_random
        }
        print(metric)
        
        if not (args.model_quality_path is None):
            json.dump(metric, open(args.model_quality_path, 'w'))
        
    else:
        raise Exception(f'Unrecognized model type {args.model_type}')