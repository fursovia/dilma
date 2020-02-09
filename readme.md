# Adversarial Attacks (ADAT)

## Datasets

**How every dataset should look like:**
1. .csv format (train.csv and test.csv)
2. Two columns `[sequence, label]` (order is important!)
3. `sequence` is a `string`, where each event is separated by a space.
4. `label` is an `int`.

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
We need two models to use MCMC sampler: classification model and seq2seq model (encoder-decoder like).

To train these models run the following commands

```bash
python train.py \
    --task att_mask_seq2seq \
    --model_dir experiments/tinkoff/att_mask_seq2seq \
    --data_dir data/tinkoff \
    --use_mask \
    -ne 20 \
    --cuda 1
```

and

```bash
python train.py \
    --task classification \
    --model_dir experiments/tinkoff/classification \
    --data_dir data/tinkoff \
    -ne 30 \
    --cuda 1
```


```bash
python train.py \
    --task classification_seq2seq \
    --model_dir experiments/tinkoff/classification_att_mask_seq2seq_fixed \
    --data_dir data/tinkoff \
    --seq2seq_model_dir experiments/tinkoff/att_mask_seq2seq \
    -ne 30 \
    --cuda 1
```

Run `python train.py --help` to see all available arguments


### Deep Levenshtein

```bash
python train.py \
    --task deep_levenshtein \
    --model_dir experiments/tinkoff/deep_levenshtein \
    --data_dir data/deep_lev_tinkoff \
    -ne 30 \
    --cuda 2
```


```bash
python train.py \
    --task deep_levenshtein_att \
    --model_dir experiments/tinkoff/deep_levenshtein_att \
    --data_dir data/deep_lev_tinkoff \
    -ne 30 \
    --cuda 2
```


```bash
python train.py \
    --task deep_levenshtein_seq2seq \
    --model_dir experiments/tinkoff/deep_levenshtein_seq2seq \
    --data_dir data/deep_lev_tinkoff \
    -ne 30 \
    --seq2seq_model_dir experiments/tinkoff/att_mask_seq2seq \
    --cuda 1
```


### Adversarial examples

```bash
python run_mcmc.py \
    --csv_path data/kaggle_transactions_data/test.csv \
    --results_path results_last \
    --class_dir experiments/classification_exp_43 \
    --seq2seq_dir experiments/seq2seq_exp_new_mask \
    --cuda 0
```


### Gradient-based adversarial examples

```bash
python run_gradient_attack.py \
    --csv_path data/tinkoff/test.csv \
    --results_path results_tinkoff \
    --seq2seq_path experiments/tinkoff/att_mask_seq2seq \
    --classification_path experiments/tinkoff/classification_att_mask_seq2seq_fixed \
    --levenshtein_path experiments/tinkoff/deep_levenshtein_seq2seq \
    --cuda 0
```