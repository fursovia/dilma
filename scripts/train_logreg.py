import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

from adat.models.classification_model import LogisticRegressionOnTfIdf

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, help='path to dataset with train.csv and test.csv files')
parser.add_argument('--model_dir', type=str, help='path where to store classifier')


if __name__ == '__main__':
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(exist_ok=True, parents=True)

    train_data = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
    test_data = pd.read_csv(os.path.join(args.data_dir, 'test.csv'))

    train_inds = np.random.choice(len(train_data), int(0.7 * len(train_data)), replace=False)
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
            f1 = f1_score(test_y, (probs * w).argmax(axis=1), average='macro')
        else:
            f1 = f1_score(test_y, (probs * w).argmax(axis=1))
        f1s.append(f1)

    best_w = ws[np.argmax(f1s)]

    # fit full model
    train_x = train_data.sequences.values
    train_y = train_data.labels.values

    test_x = test_data.sequences.values
    test_y = test_data.labels.values

    model = LogisticRegressionOnTfIdf()
    model.fit(train_x, train_y)

    joblib.dump({'model': model, 'weights': best_w}, model_dir / 'model')

    probs = model.predict(test_x)

    if len(set(test_y)) > 2:
        auc = roc_auc_score(y_true=test_y, y_score=probs, multi_class='ovr', average='macro')
    else:
        auc = roc_auc_score(y_true=test_y, y_score=probs[:, 1])
    acc = (test_y == (probs * best_w).argmax(axis=1)).mean()
    acc_random = (test_y == np.random.choice(test_y, len(test_y), replace=False)).mean()

    metric = {
        'ROC AUC': auc,
        'Accuracy': acc,
        'Accuracy(random)': acc_random
    }
    print(metric)

    json.dump(metric, open(model_dir / 'metrics.json', 'w'))

