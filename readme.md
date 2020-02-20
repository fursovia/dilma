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


```bash
# bash bin/train_models_for_cascada.sh {DATA_DIR} {GPU_ID} {NUM_CLASSES} {COPYNET_TYPE} {EXP_DIR}
bash bin/train_models_for_cascada.sh data/my_data 0 2 masked_copynet_with_attention experiments/my_experiment
```