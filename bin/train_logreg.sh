#!/usr/bin/env bash

# bash bin/train_logreg.sh {DATA_DIR}

DATA_DIR=$1
MODEL_DIR=experiments/$(basename ${DATA_DIR})/logreg

python train_logreg.py --data_dir ${DATA_DIR} --model_dir ${MODEL_DIR}
