#!/usr/bin/env bash

# usage
# bash bin/train_models.sh {DATA_DIR} {LOG_DIR} {NUM_CLASSES}

DATA_DIR=$1
LOG_DIR=$2
NUM_CLASSES=$3

LM_DATA_DIR=${DATA_DIR}/lm
ORIG_CLS_DATA_DIR=${DATA_DIR}/original_class
SUB_CLS_DATA_DIR=${DATA_DIR}/substitute_class
DL_DATA_DIR=${DATA_DIR}/lev

mkdir -p ${LOG_DIR}
mkdir -p ${DL_DATA_DIR}


echo ">>>>>>> Training Language Model"
export LM_TRAIN_DATA_PATH=${LM_DATA_DIR}/train.json
export LM_VALID_DATA_PATH=${LM_DATA_DIR}/test.json

allennlp train configs/lm/transformer_masked_lm.jsonnet \
    -s ${LOG_DIR}/lm \
    --include-package adat


echo ">>>>>>> Training Target Classifier"
export CLS_TRAIN_DATA_PATH=${ORIG_CLS_DATA_DIR}/train.json
export CLS_VALID_DATA_PATH=${ORIG_CLS_DATA_DIR}/test.json
export LM_VOCAB_PATH=${LOG_DIR}/lm/vocabulary
export CLS_NUM_CLASSES=${NUM_CLASSES}

allennlp train configs/classifier/gru_classifier.jsonnet \
    -s ${LOG_DIR}/original_class_gru \
    --include-package adat


echo ">>>>>>> Training Substitute Classifier"
export CLS_TRAIN_DATA_PATH=${SUB_CLS_DATA_DIR}/train.json
export CLS_VALID_DATA_PATH=${SUB_CLS_DATA_DIR}/test.json
export LM_VOCAB_PATH=${LOG_DIR}/lm/vocabulary
export CLS_NUM_CLASSES=${NUM_CLASSES}

allennlp train configs/classifier/gru_classifier.jsonnet \
    -s ${LOG_DIR}/substitute_class_gru \
    --include-package adat


if [ -f "$DL_TRAIN_DATA_PATH" ]; then
    echo "Skipping Levenshtein dataset creation"
else
    PYTHONPATH=. python scripts/create_levenshtein_dataset.py \
        --data-dir ${LM_DATA_DIR} \
        --output-dir ${DL_DATA_DIR}
fi


echo ">>>>>>> Training Deep Levenshtein"
export DL_TRAIN_DATA_PATH=${DL_DATA_DIR}/train.json
export DL_VALID_DATA_PATH=${DL_DATA_DIR}/test.json
allennlp train configs/levenshtein/cnn_deep_levenshtein.jsonnet \
    -s ${LOG_DIR}/lev \
    --include-package adat
