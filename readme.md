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
    --task seq2seq \
    --model_dir experiments/seq2seq_exp_maskers \
    --data_dir data/kaggle_transactions_data \
    --use_mask \
    -ne 20 \
    --cuda 3
```

and

```bash
python train.py \
    --task classification \
    --model_dir experiments/classification_exp_43 \
    --data_dir data/kaggle_transactions_data \
    --cuda 0
```

Run `python train.py --help` to see all available arguments

### Adversarial examples

```bash
python run_mcmc.py \
    --csv_path data/kaggle_transactions_data/test.csv \
    --results_path results_last \
    --class_dir experiments/classification_exp_43 \
    --seq2seq_dir experiments/seq2seq_exp_new_mask \
    --cuda 0
```