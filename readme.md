# Adversarial Attacks (ADAT)

## Dependencies

Install [Poetry](https://python-poetry.org/)

```bash
pip install poetry==0.12.17
```

Install all project dependencies

```bash
poetry install
```

And active a virtual environment

```bash
poetry shell
```

## Training

Masked LM

```bash
export LM_TRAIN_DATA_PATH="data/wine_lm/train.json"
export LM_VALID_DATA_PATH="data/wine_lm/test.json"

allennlp train configs/lm/transformer_masked_lm.jsonnet \
    -s logs/wine_lm \
    --include-package adat
```

Classifier

```bash
export CLS_TRAIN_DATA_PATH="data/ag_news_class/train.json"
export CLS_VALID_DATA_PATH="data/ag_news_class/test.json"
export LM_VOCAB_PATH="logs/ag_news_models/lm/vocabulary"
export CLS_NUM_CLASSES=4

allennlp train configs/classifier/cnn_classifier.jsonnet \
    -s logs/ag_news_models/cnn_classifier_2 \
    --include-package adat
```

Deep Levenshtein

```bash
export DL_TRAIN_DATA_PATH="data/ag_news/levenshtein/train.json"
export DL_VALID_DATA_PATH="data/ag_news/levenshtein/test.json"
export LM_VOCAB_PATH="logs/ag_news_models/lm/vocabulary"

allennlp train configs/levenshtein/cnn_deep_levenshtein.jsonnet \
    -s logs/ag_news_models/lev \
    --include-package adat
```


## Adversarial Attacks

CASCADA

```bash
PYTHONPATH=. python scripts/cascada_attack.py \
    --config-path configs/cascada/random_sampling_config.json \
    --lm-dir logs/ag_news_models/lm/ \
    --classifier-dir logs/ag_news_models/cnn_classifier/ \
    --deep-levenshtein-dir logs/ag_news_models/levenshtein_full_2/ \
    --test-path data/ag_news_class/train.json \
    --out-dir results/cascada/ \
    --sample-size 10000 \
    --cuda 0
```

HotFlip

```bash
PYTHONPATH=. python scripts/hotflip_attack.py \
    --classifier-dir logs/ag_news_models/cnn_classifier/ \
    --test-path data/ag_news_class/train.json \
    --out-dir results/discr/hotflip \
    --sample-size 10000 \
    --not-date-dir \
    --cuda 0
```

## Evaluate attacks

```bash
PYTHONPATH=. python scripts/evaluate_attack.py \
    --adversarial-dir results/hotflip/20200509_183543 \
    --classifier-dir logs/ag_news_models/gru_classifier_ft2
```

## Fine-tune

Prepare dataset

```bash
PYTHONPATH=. python scripts/prepare_for_fine_tuning.py \
    --adversarial-dir results/hotflip/20200509_182410 \
    --mix-with-path data/ag_news_class/train.json
```

Fine-tune
```bash
PYTHONPATH=. python scripts/fine_tune.py \
    --log-dir logs/ag_news_models/gru_classifier \
    --fine-tune-dir logs/ag_news_models/gru_classifier_ft2 \
    --train-path results/hotflip/20200509_182410/fine_tuning_data.json \
    --cuda 0
```


```bash
allennlp evaluate \
    logs/ag_news_models/gru_classifier_ft2/model.tar.gz \
    results/hotflip/20200509_182410/fine_tuning_data.json \
    --include-package adat
```