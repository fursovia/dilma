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
export LM_TRAIN_DATA_PATH="data/nlp_lm/train.json"
export LM_VALID_DATA_PATH="data/nlp_lm/test.json"
allennlp train configs/lm/transformer_masked_lm.jsonnet \
    -s logs/nlp_lm \
    --include-package adat
```

AR LM

```bash
export LM_TRAIN_DATA_PATH="data/nlp_lm/train.json"
export LM_VALID_DATA_PATH="data/nlp_lm/test.json"
allennlp train configs/ar_lm/lstm_lm.jsonnet \
    -s logs/ar_nlp_lm \
    --include-package adat --include-package allennlp_models
```

Classifier

```bash
export CLS_TRAIN_DATA_PATH="data/ag_news_class/train.json"
export CLS_VALID_DATA_PATH="data/ag_news_class/test.json"
export LM_VOCAB_PATH="logs/ag_news_models/lm/vocabulary"
export LM_ARCHIVE_PATH="logs/ag_news_models/lm/model.tar.gz"
export CLS_NUM_CLASSES=4

allennlp train configs/distribution_classifier/cnn_distribution_classifier.jsonnet \
    -s logs/ag_news_models/cnn_distribution_classifier \
    --include-package adat
```

Distribution Classifier

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
export DL_TRAIN_DATA_PATH="datasets/nlp_lev/train.json"
export DL_VALID_DATA_PATH="datasets/nlp_lev/test.json"
export LM_VOCAB_PATH="logs/nlp_lm/vocabulary"

allennlp train configs/levenshtein/cnn_deep_levenshtein.jsonnet \
    -s logs/nlp_lev \
    --include-package adat
```

Distribution Deep Levenshtein

```bash
export DL_TRAIN_DATA_PATH="datasets/nlp_lev/train.json"
export DL_VALID_DATA_PATH="datasets/nlp_lev/test.json"
export LM_VOCAB_PATH="logs/nlp_lm/vocabulary"
eexport LM_ARCHIVE_PATH="logs/nlp_lm/model.tar.gz"

allennlp train configs/distribution_levenshtein/cnn_distribution_deep_levenshtein.jsonnet \
    -s logs/nlp_lm \
    --include-package adat
```


## Adversarial Attacks

CASCADA

```bash
PYTHONPATH=. python scripts/cascada_attack.py \
    --config-path configs/distribution_cascada/base_config.json \
    --lm-dir logs/ag_news_models/lm/ \
    --classifier-dir logs/ag_news_models/cnn_distribution_classifier/ \
    --deep-levenshtein-dir logs/ag_news_models/distribution_lev/ \
    --test-path data/ag_news_class/test.json \
    --out-dir results/cascada/ \
    --sample-size 250 \
    --distribution-level \
    --cuda 0
```

HotFlip

```bash
PYTHONPATH=. python scripts/hotflip_attack.py \
    --classifier-dir logs/imdb/substitute_class_gru/ \
    --test-path data/imdb/original_class/test.json \
    --out-dir results/hotflip \
    --sample-size 500 \
    --cuda 0
```

## Evaluate attacks

```bash
PYTHONPATH=. python scripts/evaluate_attack.py \
    --adversarial-dir results/hotflip/20200513_220551 \
    --classifier-dir logs/imdb/original_class_gru
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


## Discriminator

Prepare dataset
```bash
PYTHONPATH=. python scripts/prepare_for_discr.py \
    --adversarial-dir results/discr/cascada \
    --out-dir results/discr/cascada
```

Discriminator

```bash
export DISCR_TRAIN_DATA_PATH="results/discr/cascada/train.json"
export DISCR_VALID_DATA_PATH="results/discr/cascada/test.json"

allennlp train configs/classifier/gru_discriminator.jsonnet \
    -s logs/ag_news_models/cascada_discriminator \
    --include-package adat
```