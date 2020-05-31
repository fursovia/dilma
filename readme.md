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

This script will output

|    | dataset   | adversary        |   NAD_1.0 |   mean_prob_diff |   mean_wer |
|---:|:----------|:-----------------|----------:|-----------------:|-----------:|
|  0 | ag        | hotflip          |      0.75 |             0.62 |       1.26 |
|  1 | ag        | cascada          |      0.45 |             0.56 |       2.05 |
|  2 | ag        | cascada_sampling |      0.58 |             0.55 |       1.76 |
|  3 | ag        | sampling_fool    |      0.5  |             0.36 |       1.32 |
|  4 | mr        | hotflip          |      0.64 |             0.26 |       1.04 |
|  5 | mr        | cascada          |      0.42 |             0.3  |       2.21 |
|  6 | mr        | cascada_sampling |      0.46 |             0.29 |       1.97 |
|  7 | mr        | sampling_fool    |      0.36 |             0.17 |       2    |
|  8 | sst       | hotflip          |      0.86 |             0.73 |       0.98 |
|  9 | sst       | cascada          |      0.59 |             0.67 |       1.84 |
| 10 | sst       | cascada_sampling |      0.65 |             0.7  |       1.66 |
| 11 | sst       | sampling_fool    |      0.62 |             0.52 |       1.22 |
| 12 | trec      | hotflip          |      0.72 |             0.54 |       1.25 |
| 13 | trec      | cascada          |      0.51 |             0.5  |       1.74 |
| 14 | trec      | cascada_sampling |      0.58 |             0.5  |       1.58 |
| 15 | trec      | sampling_fool    |      0.41 |             0.26 |       1.62 |
| 16 | age       | hotflip          |      0.72 |             0.18 |       1.6  |
| 17 | age       | cascada          |      0.01 |             0    |       0.03 |
| 18 | age       | cascada_sampling |      0.7  |             0.08 |       0.97 |
| 19 | age       | sampling_fool    |      0.82 |             0.14 |       1.17 |
| 20 | gender    | hotflip          |      0.98 |             0.47 |       1    |
| 21 | gender    | cascada          |      0.11 |             0.05 |       0.41 |
| 22 | gender    | cascada_sampling |      0.67 |             0.2  |       1.3  |
| 23 | gender    | sampling_fool    |      0.75 |             0.24 |       1.2  |
| 24 | ins       | hotflip          |      0.38 |             0.61 |       2    |
| 25 | ins       | cascada          |      0.03 |             0.18 |       3.64 |
| 26 | ins       | cascada_sampling |      0.05 |             0.21 |       3.91 |
| 27 | ins       | sampling_fool    |      0.02 |             0.04 |       1.27 |


## Adversarial Training

TODO

## Adversarial Example Detection

TODO