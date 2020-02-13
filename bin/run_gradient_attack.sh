#!/usr/bin/env bash


# usage
# sh bin/train_models.sh {TASK_NAME} {MASKERS} {GPU_ID}

TASK_NAME=$1
MASKERS=$2
GPU_ID=${3:-0}
SAMPLE=10000
RESULTS_DIR=results/${TASK_NAME}/gradient/$(date +%Y%m%d_%H%M%S)
mkdir ${RESULTS_DIR}

python run_gradient_attack.py \
    --csv_path data/${TASK_NAME}/test.csv \
    --results_path ${RESULTS_DIR} \
    --seq2seq_path experiments/${TASK_NAME}/seq2seq_masked_training_add \
    --classification_path experiments/${TASK_NAME}/classification_seq2seq_add \
    --levenshtein_path experiments/${TASK_NAME}/deep_levenshtein_seq2seq_add \
    --sample ${SAMPLE} \
    --maskers ${MASKERS} \
    --cuda ${GPU_ID}
