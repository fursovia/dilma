#!/usr/bin/env bash

# usage
# bash bin/train_models.sh {LOG_DIR} {TRAIN_PATH}

LOG_DIR=$1
TRAIN_PATH=$2
VALID_PATH=$3


allennlp train ${LOG_DIR}/config.json \
    -s ${LOG_DIR} \
    --include-package adat \
    --overrides {"trainer.num_epochs": 1, "trainer.patience": 1, "train_data_path": ${TRAIN_PATH}, "validation_data_path": ${VALID_PATH}, "distributed": null} \
    --recover