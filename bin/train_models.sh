#!/usr/bin/env bash

# usage
# sh bin/train_models.sh {DATA_DIR} {GPU_ID} {NUM_CLASSES} {EXP_DIR}

DATA_DIR=$1
GPU_ID=${2:-0}
NUM_CLASSES=${3:-2}
EXP_DIR=${4:-"./experiments"}
BASE_EXP_DIR=$(basename ${DATA_DIR})
NUM_EPOCHS=30

python train.py \
    --task seq2seq \
    --model_dir ${EXP_DIR}/${BASE_EXP_DIR}/seq2seq_masked_training \
    --data_dir ${DATA_DIR} \
    --use_mask \
    -ne ${NUM_EPOCHS} \
    --cuda ${GPU_ID}