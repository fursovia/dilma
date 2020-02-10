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

Masked training with attention

```bash
python train.py \
    --task seq2seq \
    --model_dir experiments/seq2seq_basic \
    --data_dir data \
    --use_mask \
    -ne 30 \
    --cuda 1
```


No masking, no attention

```bash
python train.py \
    --task seq2seq \
    --model_dir experiments/seq2seq_basic_no_attention \
    --data_dir data \
    -ne 100 \
    -p 3 \
    --no_attention \
    --cuda 2
```


#### Classification

```bash
python train.py \
    --task classification \
    --model_dir experiments/classification_fixed \
    --data_dir data \
    -ne 30 \
    --cuda 1
```

Classification on pre-trained encoder

```bash
python train.py \
    --task classification_seq2seq \
    --model_dir experiments/classification_att_mask_seq2seq_fixed \
    --data_dir data \
    --seq2seq_model_dir experiments/att_mask_seq2seq \
    -ne 30 \
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
    --model_dir experiments/deep_levenshtein_att \
    --data_dir data \
    -ne 30 \
    --cuda 2
```

On pre-trained encoder

```bash
python train.py \
    --task deep_levenshtein_seq2seq \
    --model_dir experiments/deep_levenshtein_seq2seq \
    --data_dir data \
    -ne 30 \
    --seq2seq_model_dir experiments/att_mask_seq2seq \
    --cuda 1
```


### Adversarial examples


#### MCMC

```bash
python run_mcmc.py \
    --csv_path data/random.csv \
    --results_path results/mcmc_results_no_att \
    --classification_path experiments/classification_fixed \
    --seq2seq_path experiments/seq2seq_basic_no_attention \
    --cuda 2
```


### Gradient-based adversarial examples

```bash
python run_gradient_attack.py \
    --csv_path data/test.csv \
    --results_path results \
    --seq2seq_path experiments/att_mask_seq2seq \
    --classification_path experiments/classification_att_mask_seq2seq_fixed \
    --levenshtein_path experiments/deep_levenshtein_seq2seq \
    --cuda 0
```