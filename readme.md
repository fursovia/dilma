# Adversarial Attacks (ADAT)

## Dependencies

Install [Poetry](https://python-poetry.org/)

```bash
pip install poetry
```

Run `poetry config settings.virtualenvs.create false` if you don't want to use virtual environment.


Install all project dependencies

```bash
poetry install
```

And active a virtual environment (or don't)

```bash
poetry shell
```


## Datasets

**How every dataset should look like:**
1. .csv format (train.csv and test.csv)
2. Two columns `[sequences, labels]` (order is important!)
3. `sequences` is a `string`, where each event is separated by a space.
4. `labels` is an `int`.

### [Prediction of client gender on card transactions](https://www.kaggle.com/c/python-and-analyze-data-final-project/data)

Predict a gender of a client based on his/her transactions.

Check [this](https://github.com/fursovia/adversarial_attacks/blob/master/notebooks/kaggle_dataset_preparation.ipynb)
notebook to see how the dataset was collected.

### [Ai Academy Competition](https://onti.ai-academy.ru/competition)

Predict an age of a client based on his/her transactions.

### Insurance dataset by Martin

TODO


## Basic usage

### Training

Models needed for an attacker:
* Seq2seq model (encoder-decoder like)
* Classification model
* Deep Levenshtein model (optional)

#### Seq2seq

No masking, no attention

```bash
python train.py \
    --task seq2seq \
    --model_dir experiments/seq2seq_basic \
    --data_dir data \
    -ne 100 \
    -p 3 \
    --no_attention \
    --cuda 2
```

Masking, no attention
```bash
python train.py \
    --task seq2seq \
    --model_dir experiments/ag_news/seq2seq_masked_training_no_attention \
    --data_dir data/ag_news \
    --use_mask \
    --no_attention \
    -ne 100 \
    -p 3 \
    -bs 512 \
    --cuda 3
```


Masked training with attention

```bash
python train.py \
    --task seq2seq \
    --model_dir experiments/ag_news/seq2seq_masked_training \
    --data_dir data/ag_news \
    --use_mask \
    -ne 30 \
    --cuda 3
```


#### Classification

```bash
python train.py \
    --task classification \
    --model_dir experiments/ag_news/classification \
    --data_dir data/ag_news \
    -ne 30 \
    --num_classes 4 \
    --cuda 0
```

Classification on pre-trained encoder

```bash
python train.py \
    --task classification_seq2seq \
    --model_dir experiments/ag_news/classification_seq2seq\
    --data_dir data/ag_news \
    --seq2seq_model_dir experiments/ag_news/seq2seq_masked_training \
    -ne 30 \
    --num_classes 4 \
    --cuda 1
```


#### Deep Levenshtein

How to create a dataset: `notebooks/prepare_lev_dataset.ipynb`


Almost like in the paper

```bash
python train.py \
    --task deep_levenshtein \
    --model_dir experiments/deep_levenshtein \
    --data_dir data \
    -ne 30 \
    --cuda 2
```

With attention

```bash
python train.py \
    --task deep_levenshtein_att \
    --model_dir experiments/ag_news/deep_levenshtein_att \
    --data_dir data/ag_news_lev \
    -ne 30 \
    --cuda 2
```

On pre-trained encoder

```bash
python train.py \
    --task deep_levenshtein_seq2seq \
    --model_dir experiments/ag_news/deep_levenshtein_seq2seq \
    --data_dir data/ag_news_lev \
    -ne 30 \
    --seq2seq_model_dir experiments/ag_news/seq2seq_masked_training \
    --cuda 1
```


### Adversarial examples


#### MCMC

```bash
python run_mcmc.py \
    --csv_path data/ag_news/test.csv \
    --results_path results/ag_news/ \
    --classification_path experiments/ag_news/classification \
    --seq2seq_path experiments/ag_news/seq2seq_masked_training \
    --cuda 1
```


### Gradient-based adversarial examples

```bash
python run_gradient_attack.py \
    --csv_path data/ag_news/test.csv \
    --results_path results/ag_news/gradient \
    --seq2seq_path experiments/ag_news/seq2seq_masked_training \
    --classification_path experiments/ag_news/classification_seq2seq \
    --levenshtein_path experiments/ag_news/deep_levenshtein_seq2seq \
    --num_classes 4 \
    --cuda 1
```