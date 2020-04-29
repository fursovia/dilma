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

allennlp train training_config/lm/transformer_masked_lm.jsonnet \
    -s logs/wine_lm \
    --include-package adat
```

Classifier

```bash
export CLS_TRAIN_DATA_PATH="data/wine_class/train.json"
export CLS_VALID_DATA_PATH="data/wine_class/test.json"
export LM_VOCAB_PATH="logs/wine_lm/vocabulary"

allennlp train training_config/classifier/cnn_classifier.jsonnet \
    -s logs/wine_classifier \
    --include-package adat
```

Deep Levenshtein

```bash
export DL_TRAIN_DATA_PATH="data/wine_lev/train.json"
export DL_VALID_DATA_PATH="data/wine_lev/test.json"
export LM_VOCAB_PATH="logs/wine_lm/vocabulary"

allennlp train training_config/levenshtein/cnn_deep_levenshtein.jsonnet \
    -s logs/wine_levenshtein \
    --include-package adat
```