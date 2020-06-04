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

This script will output [test 0.2]

|    | dataset   | adversary        |   NAD_1.0 |   mean_prob_diff |   mean_wer |   misclassification_error |
|---:|:----------|:-----------------|----------:|-----------------:|-----------:|--------------------------:|
|  0 | ag        | hotflip          |      0.75 |             0.62 |       1.26 |                      0.83 |
|  1 | ag        | cascada          |      0.45 |             0.56 |       2.05 |                      0.78 |
|  2 | ag        | cascada_sampling |      0.58 |             0.55 |       1.76 |                      0.79 |
|  3 | ag        | sampling_fool    |      0.5  |             0.36 |       1.32 |                      0.56 |
|  4 | mr        | hotflip          |      0.64 |             0.26 |       1.04 |                      0.65 |
|  5 | mr        | cascada          |      0.42 |             0.3  |       2.21 |                      0.72 |
|  6 | mr        | cascada_sampling |      0.46 |             0.29 |       1.97 |                      0.68 |
|  7 | mr        | sampling_fool    |      0.36 |             0.17 |       2    |                      0.51 |
|  8 | sst       | hotflip          |      0.86 |             0.73 |       0.98 |                      0.88 |
|  9 | sst       | cascada          |      0.59 |             0.67 |       1.84 |                      0.85 |
| 10 | sst       | cascada_sampling |      0.65 |             0.7  |       1.66 |                      0.85 |
| 11 | sst       | sampling_fool    |      0.62 |             0.52 |       1.22 |                      0.67 |
| 12 | trec      | hotflip          |      0.72 |             0.54 |       1.25 |                      0.81 |
| 13 | trec      | cascada          |      0.51 |             0.5  |       1.74 |                      0.79 |
| 14 | trec      | cascada_sampling |      0.58 |             0.5  |       1.58 |                      0.76 |
| 15 | trec      | sampling_fool    |      0.41 |             0.26 |       1.62 |                      0.52 |
| 16 | age       | hotflip          |      0.72 |             0.18 |       1.6  |                      0.92 |
| 17 | age       | cascada          |      0.01 |             0    |       0.03 |                      0.57 |
| 18 | age       | cascada_sampling |      0.7  |             0.08 |       0.97 |                      0.74 |
| 19 | age       | sampling_fool    |      0.82 |             0.14 |       1.17 |                      0.87 |
| 20 | gender    | hotflip          |      0.98 |             0.47 |       1    |                      0.98 |
| 21 | gender    | cascada          |      0.11 |             0.05 |       0.41 |                      0.43 |
| 22 | gender    | cascada_sampling |      0.67 |             0.2  |       1.3  |                      0.72 |
| 23 | gender    | sampling_fool    |      0.75 |             0.24 |       1.2  |                      0.79 |
| 24 | ins       | hotflip          |      0.38 |             0.61 |       2    |                      0.79 |
| 25 | ins       | cascada          |      0.03 |             0.18 |       3.64 |                      0.14 |
| 26 | ins       | cascada_sampling |      0.05 |             0.21 |       3.91 |                      0.17 |
| 27 | ins       | sampling_fool    |      0.02 |             0.04 |       1.27 |                      0.03 |


FGSM, DeepFool part [test 0.2]

|    | dataset   | adversary   |   NAD_1.0 |   mean_prob_diff |   mean_wer |   misclassification_error |
|---:|:----------|:------------|----------:|-----------------:|-----------:|--------------------------:|
|  0 | ag        | fgsm        |      0.66 |             0.43 |       1.01 |                      0.67 |
|  1 | ag        | deepfool    |      0.48 |             0.34 |       0.93 |                      0.52 |
|  2 | mr        | fgsm        |      0.57 |             0.23 |       1.01 |                      0.58 |
|  3 | mr        | deepfool    |      0.52 |             0.19 |       0.96 |                      0.53 |
|  4 | sst       | fgsm        |      0.63 |             0.4  |       1.02 |                      0.64 |
|  5 | sst       | deepfool    |      0.59 |             0.35 |       1.01 |                      0.59 |
|  6 | trec      | fgsm        |      0.62 |             0.42 |       0.86 |                      0.62 |
|  7 | trec      | deepfool    |      0.42 |             0.27 |       0.79 |                      0.46 |
|  8 | age       | fgsm        |      0.19 |             0.02 |       0.32 |                      0.57 |
|  9 | age       | deepfool    |      0.43 |             0.07 |       0.69 |                      0.65 |
| 10 | gender    | fgsm        |      0.92 |             0.42 |       1    |                      0.92 |
| 11 | gender    | deepfool    |      0.72 |             0.33 |       0.89 |                      0.76 |
| 12 | ins       | fgsm        |      0.18 |             0.36 |       1.42 |                      0.33 |
| 13 | ins       | deepfool    |      0.18 |             0.39 |       1.42 |                      0.34 |


[valid 5.0]

|    | dataset   | adversary   |   NAD_1.0 |   mean_prob_diff |   mean_wer |   misclassification_error |
|---:|:----------|:------------|----------:|-----------------:|-----------:|--------------------------:|
|  0 | ag        | fgsm        |      0.66 |             0.46 |       1.01 |                      0.67 |
|  1 | ag        | deepfool    |      0.52 |             0.37 |       0.97 |                      0.55 |
|  2 | mr        | fgsm        |      0.58 |             0.23 |       1    |                      0.59 |
|  3 | mr        | deepfool    |      0.53 |             0.21 |       0.98 |                      0.54 |
|  4 | sst       | fgsm        |      0.66 |             0.63 |       1.24 |                      0.76 |
|  5 | sst       | deepfool    |      0.6  |             0.58 |       1.18 |                      0.7  |
|  6 | trec      | fgsm        |      0.58 |             0.34 |       0.98 |                      0.59 |
|  7 | trec      | deepfool    |      0.47 |             0.27 |       0.92 |                      0.51 |
|  8 | age       | fgsm        |      0.18 |             0.02 |       0.33 |                      0.57 |
|  9 | age       | deepfool    |      0.43 |             0.07 |       0.7  |                      0.64 |
| 10 | gender    | fgsm        |      0.92 |             0.42 |       1    |                      0.92 |
| 11 | gender    | deepfool    |      0.67 |             0.3  |       0.85 |                      0.74 |
| 12 | ins       | fgsm        |      0.17 |             0.34 |       1.42 |                      0.31 |
| 13 | ins       | deepfool    |      0.19 |             0.38 |       1.42 |                      0.35 |


## Adversarial Training

```bash
bash bin/adv_training.sh
```

|    | dataset   | adversary   |   num_adv_examples |   NAD_1.0 |   mean_prob_diff |   mean_wer |   misclassification_error |
|---:|:----------|:------------|-------------------:|----------:|-----------------:|-----------:|--------------------------:|
|  0 | ag        | fgsm        |               5000 |      0.26 |             0.17 |       1.01 |                      0.26 |
|  1 | ag        | fgsm        |                 50 |      0.66 |             0.46 |       1.01 |                      0.66 |
|  2 | ag        | fgsm        |                100 |      0.65 |             0.46 |       1.01 |                      0.66 |
|  3 | ag        | fgsm        |                500 |      0.64 |             0.45 |       1.01 |                      0.65 |
|  4 | ag        | fgsm        |               1000 |      0.62 |             0.43 |       1.01 |                      0.63 |
|  5 | ag        | deepfool    |               5000 |      0.24 |             0.17 |       0.97 |                      0.26 |
|  6 | ag        | deepfool    |                 50 |      0.51 |             0.37 |       0.97 |                      0.54 |
|  7 | ag        | deepfool    |                100 |      0.5  |             0.37 |       0.97 |                      0.54 |
|  8 | ag        | deepfool    |                500 |      0.49 |             0.36 |       0.97 |                      0.52 |
|  9 | ag        | deepfool    |               1000 |      0.48 |             0.35 |       0.97 |                      0.51 |
| 10 | mr        | fgsm        |               5000 |      0.03 |             0.01 |       1    |                      0.03 |
| 11 | mr        | fgsm        |                 50 |      0.55 |             0.21 |       1    |                      0.55 |
| 12 | mr        | fgsm        |                100 |      0.52 |             0.2  |       1    |                      0.52 |
| 13 | mr        | fgsm        |                500 |      0.25 |             0.1  |       1    |                      0.25 |
| 14 | mr        | fgsm        |               1000 |      0.03 |             0.01 |       1    |                      0.03 |
| 15 | mr        | deepfool    |               5000 |      0.03 |             0.01 |       0.98 |                      0.03 |
| 16 | mr        | deepfool    |                 50 |      0.49 |             0.19 |       0.98 |                      0.51 |
| 17 | mr        | deepfool    |                100 |      0.48 |             0.19 |       0.98 |                      0.49 |
| 18 | mr        | deepfool    |                500 |      0.24 |             0.1  |       0.98 |                      0.25 |
| 19 | mr        | deepfool    |               1000 |      0.03 |             0.01 |       0.98 |                      0.03 |
| 20 | sst       | fgsm        |               5000 |      0.1  |             0.02 |       1.24 |                      0.1  |
| 21 | sst       | fgsm        |                 50 |      0.55 |             0.51 |       1.24 |                      0.64 |
| 22 | sst       | fgsm        |                100 |      0.52 |             0.47 |       1.24 |                      0.61 |
| 23 | sst       | fgsm        |                500 |      0.33 |             0.28 |       1.24 |                      0.39 |
| 24 | sst       | fgsm        |               1000 |      0.25 |             0.2  |       1.24 |                      0.3  |
| 25 | sst       | deepfool    |               5000 |      0.08 |             0.05 |       1.18 |                      0.11 |
| 26 | sst       | deepfool    |                 50 |      0.5  |             0.47 |       1.18 |                      0.59 |
| 27 | sst       | deepfool    |                100 |      0.47 |             0.44 |       1.18 |                      0.57 |
| 28 | sst       | deepfool    |                500 |      0.3  |             0.27 |       1.18 |                      0.37 |
| 29 | sst       | deepfool    |               1000 |      0.21 |             0.18 |       1.18 |                      0.27 |
| 30 | trec      | fgsm        |               5000 |      0.02 |             0.02 |       0.98 |                      0.02 |
| 31 | trec      | fgsm        |                 50 |      0.53 |             0.34 |       0.98 |                      0.53 |
| 32 | trec      | fgsm        |                100 |      0.52 |             0.32 |       0.98 |                      0.52 |
| 33 | trec      | fgsm        |                500 |      0.03 |             0.03 |       0.98 |                      0.03 |
| 34 | trec      | fgsm        |               1000 |      0.02 |             0.02 |       0.98 |                      0.02 |
| 35 | trec      | deepfool    |               5000 |      0.01 |             0    |       0.92 |                      0.01 |
| 36 | trec      | deepfool    |                 50 |      0.43 |             0.24 |       0.92 |                      0.46 |
| 37 | trec      | deepfool    |                100 |      0.44 |             0.24 |       0.92 |                      0.47 |
| 38 | trec      | deepfool    |                500 |      0.15 |             0.11 |       0.92 |                      0.16 |
| 39 | trec      | deepfool    |               1000 |      0.01 |             0    |       0.92 |                      0.01 |

## Adversarial Example Detection

TODO