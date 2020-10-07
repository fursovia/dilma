# DILMA

This repository contains a pytorch implementation of 
["Differentiable Language Model Adversarial Attacks on Categorical Sequence Classifiers"](https://arxiv.org/abs/2006.11078).

<p align="center"><img src="imgs/dilma_architecture.png" width="480"\></p>


## Dependencies

We use [Poetry](https://python-poetry.org/) for dependency management.

```bash
pip install poetry==0.12.17
poetry install
poetry shell
```

## Datasets/Models

All datasets are available inside `dilma/dilma:latest` docker container in `/src/datasets` folder.
 Pretrained models are stored in `/src/logs`.
You can access these files by running

```bash
docker run --rm -itd --entrypoint /bin/bash \
    --name dilma \
    -v $PWD:/workspace \
    dilma/dilma:latest

docker exec -it dilma bash
```


## Attacks

Generate attacks for HotFlip, SamplingFool, DILMA (aka CASCADA), DILMA (aka CASCADA) w/ sampling by running

```bash
export GPU_ID="0"

docker run --rm --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=$GPU_ID \
    --entrypoint /bin/bash \
    dilma/dilma:latest bin/attack.sh
```

This script will output 

|    | dataset   | adversary        |   NAD_1.0 |   mean_prob_diff |   mean_wer |   misclassification_error |
|---:|:----------|:-----------------|----------:|-----------------:|-----------:|--------------------------:|
|  0 | ag        | fgsm             |      0.65 |             0.46 |       1.01 |                      0.66 |
|  1 | ag        | deepfool         |      0.5  |             0.37 |       0.97 |                      0.53 |
|  2 | ag        | hotflip          |      0.74 |             0.62 |       1.26 |                      0.84 |
|  3 | ag        | cascada          |      0.41 |             0.48 |       2.1  |                      0.67 |
|  4 | ag        | cascada_sampling |      0.53 |             0.52 |       1.74 |                      0.71 |
|  5 | ag        | sampling_fool    |      0.44 |             0.34 |       1.37 |                      0.5  |
|  6 | mr        | fgsm             |      0.57 |             0.23 |       1    |                      0.57 |
|  7 | mr        | deepfool         |      0.53 |             0.2  |       0.98 |                      0.54 |
|  8 | mr        | hotflip          |      0.62 |             0.27 |       1.04 |                      0.63 |
|  9 | mr        | cascada          |      0.38 |             0.28 |       2.62 |                      0.66 |
| 10 | mr        | cascada_sampling |      0.45 |             0.27 |       1.97 |                      0.63 |
| 11 | mr        | sampling_fool    |      0.35 |             0.16 |       1.96 |                      0.47 |
| 12 | sst       | fgsm             |      0.67 |             0.64 |       1.24 |                      0.77 |
| 13 | sst       | deepfool         |      0.6  |             0.58 |       1.18 |                      0.7  |
| 14 | sst       | hotflip          |      0.81 |             0.69 |       0.96 |                      0.83 |
| 15 | sst       | cascada          |      0.52 |             0.6  |       1.91 |                      0.75 |
| 16 | sst       | cascada_sampling |      0.63 |             0.63 |       1.54 |                      0.78 |
| 17 | sst       | sampling_fool    |      0.57 |             0.48 |       1.24 |                      0.61 |
| 18 | trec      | fgsm             |      0.56 |             0.32 |       0.98 |                      0.56 |
| 19 | trec      | deepfool         |      0.46 |             0.27 |       0.92 |                      0.5  |
| 20 | trec      | hotflip          |      0.69 |             0.53 |       1.29 |                      0.78 |
| 21 | trec      | cascada          |      0.56 |             0.52 |       1.53 |                      0.78 |
| 22 | trec      | cascada_sampling |      0.72 |             0.56 |       1.32 |                      0.83 |
| 23 | trec      | sampling_fool    |      0.41 |             0.28 |       1.64 |                      0.51 |
| 24 | age       | fgsm             |      0.18 |             0.02 |       0.33 |                      0.57 |
| 25 | age       | deepfool         |      0.42 |             0.07 |       0.7  |                      0.63 |
| 26 | age       | hotflip          |      0.69 |             0.2  |       1.67 |                      0.9  |
| 27 | age       | cascada          |      0.57 |             0.16 |       2.28 |                      0.87 |
| 28 | age       | cascada_sampling |      0.78 |             0.16 |       1.51 |                      0.9  |
| 29 | age       | sampling_fool    |      0.84 |             0.15 |       1.12 |                      0.87 |
| 30 | gender    | fgsm             |      0.92 |             0.42 |       1    |                      0.93 |
| 31 | gender    | deepfool         |      0.67 |             0.3  |       0.86 |                      0.74 |
| 32 | gender    | hotflip          |      0.97 |             0.48 |       1.01 |                      0.98 |
| 33 | gender    | cascada          |      0.46 |             0.25 |       2.7  |                      0.79 |
| 34 | gender    | cascada_sampling |      0.63 |             0.29 |       1.91 |                      0.84 |
| 35 | gender    | sampling_fool    |      0.73 |             0.27 |       1.33 |                      0.8  |
| 36 | ins       | fgsm             |      0.17 |             0.34 |       1.42 |                      0.31 |
| 37 | ins       | deepfool         |      0.19 |             0.38 |       1.42 |                      0.35 |
| 38 | ins       | hotflip          |      0.36 |             0.61 |       1.91 |                      0.75 |
| 39 | ins       | cascada          |      0.05 |             0.28 |       3.98 |                      0.21 |
| 40 | ins       | cascada_sampling |      0.08 |             0.35 |       4.06 |                      0.31 |
| 41 | ins       | sampling_fool    |      0.02 |             0.04 |       1.28 |                      0.02 |

## Adversarial Training

```bash
export GPU_ID="0"

docker run --rm --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=$GPU_ID \
    --entrypoint /bin/bash \
    dilma/dilma:latest bin/adv_training.sh
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
| 40 | age       | fgsm        |               5000 |      0.17 |             0.01 |       0.33 |                      0.56 |
| 41 | age       | fgsm        |                 50 |      0.18 |             0.02 |       0.33 |                      0.57 |
| 42 | age       | fgsm        |                100 |      0.18 |             0.02 |       0.33 |                      0.56 |
| 43 | age       | fgsm        |                500 |      0.18 |             0.02 |       0.33 |                      0.56 |
| 44 | age       | fgsm        |               1000 |      0.18 |             0.02 |       0.33 |                      0.56 |
| 45 | age       | deepfool    |               5000 |      0.4  |             0.05 |       0.7  |                      0.61 |
| 46 | age       | deepfool    |                 50 |      0.43 |             0.07 |       0.7  |                      0.64 |
| 47 | age       | deepfool    |                100 |      0.42 |             0.07 |       0.7  |                      0.63 |
| 48 | age       | deepfool    |                500 |      0.43 |             0.06 |       0.7  |                      0.64 |
| 49 | age       | deepfool    |               1000 |      0.42 |             0.06 |       0.7  |                      0.64 |
| 50 | gender    | fgsm        |               5000 |      0.59 |             0.14 |       1    |                      0.59 |
| 51 | gender    | fgsm        |                 50 |      0.86 |             0.31 |       1    |                      0.86 |
| 52 | gender    | fgsm        |                100 |      0.8  |             0.27 |       1    |                      0.8  |
| 53 | gender    | fgsm        |                500 |      0.76 |             0.22 |       1    |                      0.76 |
| 54 | gender    | fgsm        |               1000 |      0.73 |             0.2  |       1    |                      0.73 |
| 55 | gender    | deepfool    |               5000 |      0.26 |             0.01 |       0.85 |                      0.32 |
| 56 | gender    | deepfool    |                 50 |      0.6  |             0.2  |       0.85 |                      0.67 |
| 57 | gender    | deepfool    |                100 |      0.55 |             0.17 |       0.85 |                      0.62 |
| 58 | gender    | deepfool    |                500 |      0.49 |             0.13 |       0.85 |                      0.56 |
| 59 | gender    | deepfool    |               1000 |      0.42 |             0.1  |       0.85 |                      0.5  |
| 60 | ins       | fgsm        |               5000 |      0.04 |             0.22 |       1.42 |                      0.21 |
| 61 | ins       | fgsm        |                 50 |      0.12 |             0.28 |       1.42 |                      0.24 |
| 62 | ins       | fgsm        |                100 |      0.1  |             0.25 |       1.42 |                      0.22 |
| 63 | ins       | fgsm        |                500 |      0.04 |             0.21 |       1.42 |                      0.05 |
| 64 | ins       | fgsm        |               1000 |      0.03 |             0.16 |       1.42 |                      0.04 |
| 65 | ins       | deepfool    |               5000 |      0.04 |             0.2  |       1.42 |                      0.19 |
| 66 | ins       | deepfool    |                 50 |      0.13 |             0.3  |       1.42 |                      0.26 |
| 67 | ins       | deepfool    |                100 |      0.09 |             0.25 |       1.42 |                      0.19 |
| 68 | ins       | deepfool    |                500 |      0.08 |             0.22 |       1.42 |                      0.18 |
| 69 | ins       | deepfool    |               1000 |      0.04 |             0.19 |       1.42 |                      0.04 |

## Adversarial Example Detection

```bash
export GPU_ID="0"

docker run --rm --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=$GPU_ID \
    --entrypoint /bin/bash \
    dilma/dilma:latest bin/adv_detection.sh
```

|    | dataset   | adversary        |   roc_auc |
|---:|:----------|:-----------------|----------:|
|  0 | ag        | deepfool         |      0.89 |
|  1 | ag        | fgsm             |      0.96 |
|  2 | ag        | cascada          |      0.69 |
|  3 | ag        | cascada_sampling |      0.64 |
|  4 | ag        | hotflip          |      1    |
|  5 | ag        | sampling_fool    |      0.6  |
|  6 | sst       | deepfool         |      0.96 |
|  7 | sst       | fgsm             |      0.98 |
|  8 | sst       | cascada          |      0.67 |
|  9 | sst       | cascada_sampling |      0.68 |
| 10 | sst       | hotflip          |      1    |
| 11 | sst       | sampling_fool    |      0.61 |
| 12 | trec      | deepfool         |      0.75 |
| 13 | trec      | fgsm             |      0.76 |
| 14 | trec      | cascada          |      0.75 |
| 15 | trec      | cascada_sampling |      0.72 |
| 16 | trec      | hotflip          |      0.96 |
| 17 | trec      | sampling_fool    |      0.63 |
| 18 | mr        | deepfool         |      0.96 |
| 19 | mr        | fgsm             |      0.94 |
| 20 | mr        | cascada          |      0.66 |
| 21 | mr        | cascada_sampling |      0.6  |
| 22 | mr        | hotflip          |      0.98 |
| 23 | mr        | sampling_fool    |      0.59 |
| 24 | gender    | deepfool         |      0.95 |
| 25 | gender    | fgsm             |      0.99 |
| 26 | gender    | cascada          |      0.67 |
| 27 | gender    | cascada_sampling |      0.61 |
| 28 | gender    | hotflip          |      1    |
| 29 | gender    | sampling_fool    |      0.7  |
| 30 | age       | deepfool         |      0.85 |
| 31 | age       | fgsm             |      0.6  |
| 32 | age       | cascada          |      0.63 |
| 33 | age       | cascada_sampling |      0.59 |
| 34 | age       | hotflip          |      1    |
| 35 | age       | sampling_fool    |      0.66 |
| 36 | ins       | deepfool         |      0.86 |
| 37 | ins       | fgsm             |      0.96 |
| 38 | ins       | cascada          |      0.97 |
| 39 | ins       | cascada_sampling |      0.98 |
| 40 | ins       | hotflip          |      1    |
| 41 | ins       | sampling_fool    |      0.6  |


# Citation

```text
@article{fursov2020differentiable,
  title={Differentiable Language Model Adversarial Attacks on Categorical Sequence Classifiers},
  author={I. Fursov and A. Zaytsev and N. Kluchnikov and A. Kravchenko and E. Burnaev},
  journal={arXiv preprint arXiv:2006.11078},
  year={2020}
}
```