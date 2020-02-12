#!/usr/bin/env bash

# usage
# sh bin/train_models.sh {DATA_DIR} {GPU_ID} {NUM_CLASSES} {EXP_DIR}

DATA_DIR=$1
GPU_ID=${2:-0}
NUM_CLASSES=${3:-2}
EXP_DIR=${4:-"./experiments"}
BASE_EXP_DIR=$(basename ${DATA_DIR})
NUM_EPOCHS=100
PATIENCE=3

python train.py \
    --task seq2seq \
    --model_dir ${EXP_DIR}/${BASE_EXP_DIR}/seq2seq_masked_training \
    --data_dir ${DATA_DIR} \
    --use_mask \
    -ne ${NUM_EPOCHS} \
    -p ${PATIENCE} \
    --num_classes ${NUM_CLASSES} \
    --cuda ${GPU_ID}


python train.py \
    --task seq2seq \
    --model_dir ${EXP_DIR}/${BASE_EXP_DIR}/seq2seq_masked_training_no_attention \
    --data_dir ${DATA_DIR} \
    --use_mask \
    --no_attention \
    -ne ${NUM_EPOCHS} \
    -p ${PATIENCE} \
    --num_classes ${NUM_CLASSES} \
    --cuda ${GPU_ID}


python train.py \
    --task classification \
    --model_dir ${EXP_DIR}/${BASE_EXP_DIR}/classification \
    --data_dir ${DATA_DIR} \
    -ne ${NUM_EPOCHS} \
    -p ${PATIENCE} \
    --num_classes ${NUM_CLASSES} \
    --cuda ${GPU_ID}


python train.py \
    --task classification_seq2seq \
    --model_dir ${EXP_DIR}/${BASE_EXP_DIR}/classification_seq2seq \
    --data_dir ${DATA_DIR} \
    --seq2seq_model_dir ${EXP_DIR}/${BASE_EXP_DIR}/seq2seq_masked_training \
    -ne ${NUM_EPOCHS} \
    -p ${PATIENCE} \
    --num_classes ${NUM_CLASSES} \
    --cuda ${GPU_ID}