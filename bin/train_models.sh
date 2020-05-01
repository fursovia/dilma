#!/usr/bin/env bash

# usage
# bash bin/train_models.sh {LM_DATA_DIR} {CLS_DATA_DIR} {DL_DATA_DIR} {LOG_DIR} {NUM_CLASSES}

LM_DATA_DIR=$1
CLS_DATA_DIR=$2
DL_DATA_DIR=$3
LOG_DIR=$4
NUM_CLASSES=$5

export LM_TRAIN_DATA_PATH=${LM_DATA_DIR}/train.json
export LM_VALID_DATA_PATH=${LM_DATA_DIR}/test.json
export CLS_TRAIN_DATA_PATH=${CLS_DATA_DIR}/train.json
export CLS_VALID_DATA_PATH=${CLS_DATA_DIR}/test.json
export DL_TRAIN_DATA_PATH=${DL_DATA_DIR}/train.json
export DL_VALID_DATA_PATH=${DL_DATA_DIR}/test.json
export CLS_NUM_CLASSES=${NUM_CLASSES}
export LM_VOCAB_PATH=${LOG_DIR}/lm/vocabulary

mkdir -p ${LOG_DIR}
mkdir -p ${DL_DATA_DIR}

allennlp train training_config/lm/transformer_masked_lm.jsonnet \
    -s ${LOG_DIR}/lm \
    --include-package adat

allennlp train training_config/classifier/cnn_classifier.jsonnet \
    -s ${LOG_DIR}/classifier \
    --include-package adat

if [ -f "$DL_TRAIN_DATA_PATH" ]; then
    echo "Skipping Levenshtein dataset creation"
else
    PYTHONPATH=. python scripts/create_levenshtein_dataset.py \
        --data-path ${LM_DATA_DIR}/train.json \
        --output-dir ${DL_DATA_DIR}
fi

allennlp train training_config/levenshtein/cnn_deep_levenshtein.jsonnet \
    -s ${LOG_DIR}/levenshtein \
    --include-package adat
