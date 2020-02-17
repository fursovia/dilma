#!/usr/bin/env bash

# usage
# sh bin/train_models_for_cascada.sh {DATA_DIR} {GPU_ID} {NUM_CLASSES} {COPYNET_TYPE} {EXP_DIR}

DATA_DIR=$1
LEVENSHTEIN_DATA_DIR=${DATA_DIR}/levenshtein
GPU_ID=${2:-0}
NUM_CLASSES=${3:-2}
EXP_DIR=${5:-"./experiments"}
BASE_EXP_DIR=$(basename ${DATA_DIR})
BATCH_SIZE=512
NUM_EPOCHS=100
PATIENCE=4

COPYNET_TYPE=${4:-"masked_copynet_with_attention"}
CLASSIFIER_TYPE=classification_copynet
DEEP_LEVENSHTEIN_TYPE=deep_levenshtein_copynet

COPYNET_DIR=${EXP_DIR}/${BASE_EXP_DIR}/${COPYNET_TYPE}
CLASSIFIER_DIR=${EXP_DIR}/${BASE_EXP_DIR}/classifier_${COPYNET_TYPE}
LEVENSHTEIN_DIR=${EXP_DIR}/${BASE_EXP_DIR}/deep_levenshtein_${COPYNET_TYPE}


echo "Training CopyNet..."
python train.py \
    --task ${COPYNET_TYPE} \
    --model_dir ${COPYNET_DIR} \
    --data_dir ${DATA_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --patience ${PATIENCE} \
    --use_mask \
    --cuda ${GPU_ID}


echo "Training Classifier..."
python train.py \
    --task ${CLASSIFIER_TYPE} \
    --model_dir ${CLASSIFIER_DIR} \
    --data_dir ${DATA_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --patience ${PATIENCE} \
    --num_classes ${NUM_CLASSES} \
    --copynet_dir ${COPYNET_DIR} \
    --cuda ${GPU_ID}


if [ -d "${LEVENSHTEIN_DATA_DIR}" ]; then
    echo "Skipping Levenshtein data preparation"
else
    python create_levenshtein_dataset.py --csv_path ${DATA_DIR}/train.csv --output_dir ${LEVENSHTEIN_DATA_DIR}
fi

echo "Training Deep Levenshtein..."
python train.py \
    --task ${DEEP_LEVENSHTEIN_TYPE} \
    --model_dir ${LEVENSHTEIN_DIR} \
    --data_dir ${LEVENSHTEIN_DATA_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --patience ${PATIENCE} \
    --copynet_dir ${COPYNET_DIR} \
    --cuda ${GPU_ID}