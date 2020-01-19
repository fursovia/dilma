# Adversarial Attacks (ADAT)

## Datasets

**How every dataset should look like:**
1. .csv format (train.csv and test.csv)
2. Two columns `[sequence, label]` (order is important!)
3. `sequence` is a `string`, where each event is separated by a space.
4. `label` is an `int`.

### [Prediction of client gender on card transactions](https://www.kaggle.com/c/python-and-analyze-data-final-project/data)

Predict a gender of a client based on his/her transactions.

## [Ai Academy Competition](https://onti.ai-academy.ru/competition)

Predict an age of a client based on his/her transactions.

## Insurance dataset by Martin

TODO


## Basic usage

### Sequence likelihood

You can train a language model on original sequences and then calculate perplexity on adversarial examples.


To train a language model run

```bash
CUDA_VISIBLE_DEVICES="0" python train_lm.py --data_dir data --model_dir experiments --cuda 0
```

It assumes that there are `train.txt` and `test.txt` files in `data` folder.
This command will save a trained model and a vocabulary to `experiments` folder.

You can later use these files to calculate perplexity

```python
from allennlp.data.dataset_readers import SimpleLanguageModelingDatasetReader
from allennlp.data.vocabulary import Vocabulary

from adat.dataset import WhitespaceTokenizer
from adat.lm import get_basic_lm
from adat.utils import load_weights
from adat.utils import calculate_perplexity

reader = SimpleLanguageModelingDatasetReader(tokenizer=WhitespaceTokenizer())
vocab = Vocabulary.from_files('experiments/vocab')

model = get_basic_lm(vocab)
load_weights(model, 'experiments/model.th')

sequences = ['...', '...', ..., '...']
perplexity = calculate_perplexity(sequences, model, reader, vocab)
```