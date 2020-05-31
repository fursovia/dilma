#!/usr/bin/env bash

# usage
# bash bin/attack.sh {FILENAME} {NUM_SAMPLES} {RESULTS_DIR} {GPU_ID} {LOG_DIR} {DATA_DIR}

# test or valid
FILENAME=${1:-"test"}
default_sample_size=100
SAMPLE_SIZE=${2:-$default_sample_size}
RESULTS_DIR=${3:-"results"}
default_gpu_id=0
GPU_ID=${4:-$default_gpu_id}
LOG_DIR=${5:-"logs"}
DATA_DIR=${6:-"datasets"}

NUM_CONFIGS=100

echo ">> Attacking NLP models"
# NLP attacks use shared LM and DeepLev models
NLP_LOG_DIR=${LOG_DIR}/nlp
NLP_DATA_DIR=${DATA_DIR}/nlp
NLP_RESULTS_DIR=${RESULTS_DIR}/nlp

for dir in $(ls -d ${NLP_LOG_DIR}/dataset_*); do
    dataset=$(basename ${dir} | cut -d'_' -f 2)

    for i in $(seq 1 ${NUM_CONFIGS}); do
        echo ">>>>>>>>>>> 1/2: ${i}/${NUM_CONFIGS}"
        PYTHONPATH=. python scripts/cascada_attack.py \
            --config-path configs/attacks/cascada/grid_search/config_${i}.json \
            --test-path ${NLP_DATA_DIR}/${dataset}/target_clf/${FILENAME}.json \
            --classifier-dir ${dir}/substitute_clf \
            --deep-levenshtein-dir ${NLP_LOG_DIR}/lev \
            --lm-dir ${NLP_LOG_DIR}/lm \
            --out-dir ${NLP_RESULTS_DIR}/${dataset}/cascada/grid_search/${i} \
            --sample-size ${SAMPLE_SIZE} \
            --not-date-dir \
            --force \
            --cuda ${GPU_ID}
    done
done


echo ">> Attacking non-NLP models"
# non-NLP attacks use distinct LM and DeepLev models
NON_NLP_LOG_DIR=${LOG_DIR}/non_nlp
NON_NLP_DATA_DIR=${DATA_DIR}/non_nlp
NON_NLP_RESULTS_DIR=${RESULTS_DIR}/non_nlp

for dir in $(ls -d ${NON_NLP_LOG_DIR}/dataset_*); do
    dataset=$(basename ${dir} | cut -d'_' -f 2)

    for i in $(seq 1 ${NUM_CONFIGS}); do
        echo ">>>>>>>>>>> 2/2: ${i}/${NUM_CONFIGS}"
        PYTHONPATH=. python scripts/cascada_attack.py \
            --config-path configs/attacks/cascada/grid_search/config_${i}.json \
            --test-path ${NON_NLP_DATA_DIR}/${dataset}/target_clf/${FILENAME}.json \
            --classifier-dir ${dir}/substitute_clf \
            --deep-levenshtein-dir ${dir}/lev \
            --lm-dir ${dir}/lm \
            --out-dir ${NON_NLP_RESULTS_DIR}/${dataset}/cascada/grid_search/${i} \
            --sample-size ${SAMPLE_SIZE} \
            --not-date-dir \
            --force \
            --cuda ${GPU_ID}
    done
done