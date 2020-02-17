#!/usr/bin/env bash

# usage
# sh bin/train_classifier.sh {dataset} {GPU_ID} {NUM_CLASSES}


dataset=$1
DATA_DIR=data/${dataset}
GPU_ID=${2:-0}
NUM_CLASSES=${3:-2}
EXP_DIR=experiments/${dataset}
BATCH_SIZE=256
NUM_EPOCHS=30
PATIENCE=4


CLASSIFIER_DIR=${EXP_DIR}/classifier_basic

python train.py \
    --task classification \
    --model_dir ${CLASSIFIER_DIR} \
    --data_dir ${DATA_DIR} \
    --num_epochs ${NUM_EPOCHS} \
    --patience ${PATIENCE} \
    --num_classes ${NUM_CLASSES} \
    --cuda ${GPU_ID}
