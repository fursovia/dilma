import argparse
import json
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
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
        
        train_x = train_data.sequences.values
        train_y = train_data.labels.values

        test_x = test_data.sequences.values
        test_y = test_data.labels.values

        model = LogisticRegressionOnTfIdf()
        model.fit(train_x, train_y)
        
        joblib.dump(model, args.model_path)
        
        probs = model.predict(test_x)
        
        auc = roc_auc_score(y_true=test_y, y_score=probs, multi_class='ovr', average='macro')
        acc = (test_y == probs.argmax(axis=1)).mean()
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