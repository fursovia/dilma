# Adversarial Attacks (ADAT)

## Dependencies

We use [Poetry](https://python-poetry.org/) for dependency management.

```bash
pip install poetry==0.12.17
poetry install
poetry shell
```

## Training

TODO

## Attacks

Generate attacks for HotFlip, SamplingFool, CASCADA, CASCADA w/ sampling by running

```
bash bin/attack.sh 
```

Evaluate those attacks by running

```
bash bin/evaluate.sh 
```

## Adversarial Training

TODO

## Adversarial Example Detection

TODO