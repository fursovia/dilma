import argparse
from pathlib import Path
import csv
import json
from tqdm import tqdm

import pandas as pd
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.util import dump_metrics
from allennlp.predictors import TextClassifierPredictor

from adat.dataset import ClassificationReader
from adat.attackers.hotflip import HotFlipFixed
from adat.models import get_classification_model
from adat.utils import load_weights

parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', type=str, default='data/test.csv')
parser.add_argument('--results_path', type=str, default='results')
parser.add_argument('--classifier_path', type=str, default='experiments/classification')
parser.add_argument('--sample', type=int, default=None)


def _get_classifier_from_args(vocab: Vocabulary, path: str):
    with open(path) as file:
        args = json.load(file)
    num_classes = args['num_classes']
    return get_classification_model(vocab, int(num_classes))


if __name__ == '__main__':
    args = parser.parse_args()

    class_reader = ClassificationReader(skip_start_end=True)
    class_vocab = Vocabulary.from_files(Path(args.classifier_path) / 'vocab')
    class_model = _get_classifier_from_args(class_vocab, Path(args.classifier_path) / 'args.json')
    load_weights(class_model, Path(args.classifier_path) / 'best.th')

    predictor = TextClassifierPredictor(class_model, class_reader)
    attacker = HotFlipFixed(predictor)
    attacker.initialize()

    data = pd.read_csv(args.csv_path)
    sequences = data['sequences'].tolist()[:args.sample]
    labels = data['labels'].tolist()[:args.sample]

    results_path = Path(args.results_path)
    results_path.mkdir(exist_ok=True, parents=True)
    path_to_results_file = results_path / 'results.csv'
    assert not path_to_results_file.exists(), \
        f'You already have `{path_to_results_file}` file. Delete it or change --results_path.'
    dump_metrics(results_path / 'args.json', args.__dict__)
    with open(path_to_results_file, 'w', newline='') as csv_write:
        fieldnames = ['generated_sequence']
        writer = csv.DictWriter(csv_write, fieldnames=fieldnames)
        writer.writeheader()
        for seq, lab in tqdm(zip(sequences, labels)):
            inputs = {
                'sentence': seq,
                'label': lab
            }

            attack = attacker.attack_from_json(inputs)
            generated_seq = ' '.join(attack['final'][0])

            writer.writerow({'generated_sequence': generated_seq})
            csv_write.flush()
