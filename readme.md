# Adversarial Attacks (ADAT)

## Dependencies

Install [Poetry](https://python-poetry.org/)

```bash
pip install poetry
```

Run `poetry config settings.virtualenvs.create false` if you don't want to use virtual environment.


Install all project dependencies

```bash
poetry install
```

And active a virtual environment (or don't)

```bash
poetry shell
```


## Datasets

**How every dataset should look like:**
1. .csv format (train.csv and test.csv)
2. Two columns `[sequences, labels]` (order is important!)
3. `sequences` is a `string`, where each event is separated by a space.
4. `labels` is an `int`.


## Usage

What you'll need for MCMC/Random sampler:

* CopyNet (Masked or Non-Masked)
* Classifier (trained on sequenced and not on hidden representations)


What you'll need for HotFlip
* Classifier (trained on sequenced and not on hidden representations)


What you'll need for CASCADA
* CopyNet
* Classifier trained on hidden representations
* Deep Levenshtein trained on hidden representations


### Models training

```bash
# bash bin/train_models_for_cascada.sh {DATA_DIR} {GPU_ID} {NUM_CLASSES} {COPYNET_TYPE} {EXP_DIR}
bash bin/train_models_for_cascada.sh data/my_data 0 2 masked_copynet_with_attention experiments/my_experiment
```

```bash
bash bin/train_logreg.sh data/my_data
```


### Adversarial attacks

```bash
# usage
# sh bin/run_attacks.sh {DATA_DIR} {GPU_ID}
# {COPYNET_DIR} {CLASSIFIER_COPYNET_DIR} {DEEP_LEVENSHTEIN_COPYNET_DIR}
# {CLASSIFIER_BASIC_DIR}
# {BEAM_SIZE} {SAMPLE}
sh bin/run_attakcs.sh \
    data/my_data \
    3 \
    experiments/masked_copynet_with_attention experiments \
    experiments/classifier_masked_copynet_with_attention \
    experiments/deep_levenshtein_masked_copynet_with_attention \
    experiments/classifier_basic \
    3 \
    5000
```


### Evaluation

Run

```bash
python calculate_metrics.py \
    --model_path PATH_TO_LOGREG \
    --attack_results_path PATH_TO_RESULTS \
    --eval_results_path WHERE_TO_SAVE_METRICS
```
