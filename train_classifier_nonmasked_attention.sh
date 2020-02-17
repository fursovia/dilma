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

MODEL_DIR=${EXP_DIR}/classification_copynet
COPYNET_DIR=${EXP_DIR}/nonmasked_copynet_with_attention

python train.py \
    --task classification_copynet \
    --model_dir ${MODEL_DIR} \
    --data_dir ${DATA_DIR} \
    --copynet_dir ${COPYNET_DIR} \
    --num_epochs ${NUM_EPOCHS} \
    --patience ${PATIENCE} \
    --num_classes ${NUM_CLASSES} \
    --cuda ${GPU_ID}