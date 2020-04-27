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
allennlp train training_config/lm/transformer_masked_lm.jsonnet \
    -s logs/lm \
    --include-package adat
```

Classifier

```bash
allennlp train training_config/classifier/cnn_classifier.jsonnet \
    -s logs/classifier \
    --include-package adat
```

Deep Levenshtein

```bash
allennlp train training_config/levenshtein/cnn_deep_levenshtein.jsonnet \
    -s logs/levenshtein \
    --include-package adat
```