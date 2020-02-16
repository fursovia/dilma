#!/usr/bin/env bash

# usage
# sh bin/train_classifier_from_scratch.sh {DATA_DIR} {GPU_ID} {NUM_CLASSES} {EXP_DIR}

DATA_DIR=$1
GPU_ID=${2:-0}
NUM_CLASSES=${3:-2}
EXP_DIR=${4:-"./experiments"}
BASE_EXP_DIR=$(basename ${DATA_DIR})
NUM_EPOCHS=100
PATIENCE=2


python train.py \
    --task classification \
    --model_dir ${EXP_DIR}/${BASE_EXP_DIR}/classification \
    --data_dir ${DATA_DIR} \
    -ne ${NUM_EPOCHS} \
    -p ${PATIENCE} \
    --num_classes ${NUM_CLASSES} \
    --skip_start_end \
    --cuda ${GPU_ID}
