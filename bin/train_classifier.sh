#!/usr/bin/env bash

# usage
# sh bin/train_classifier.sh {DATA_DIR} {GPU_ID} {NUM_CLASSES} {EXP_DIR}

DATA_DIR=$1
GPU_ID=${2:-0}
NUM_CLASSES=${3:-2}
EXP_DIR=${4:-"./experiments"}
BASE_EXP_DIR=$(basename ${DATA_DIR})
BATCH_SIZE=512
NUM_EPOCHS=100
PATIENCE=4


CLASSIFIER_DIR=${EXP_DIR}/${BASE_EXP_DIR}/classifier_basic


python train.py \
    --task classification \
    --model_dir ${CLASSIFIER_DIR} \
    --data_dir ${DATA_DIR} \
    --num_epochs ${NUM_EPOCHS} \
    --patience ${PATIENCE} \
    --num_classes ${NUM_CLASSES} \
    --cuda ${GPU_ID}
