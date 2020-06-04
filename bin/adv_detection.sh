#!/usr/bin/env bash

# usage
# bash bin/adv_detection.sh {ATTACKS_DIR} ${GPU_ID}

set -eo pipefail -v

ATTACKS_DIR=${1:-"results"}
default_gpu_id=0
GPU_ID=${2:-$default_gpu_id}

LOGS_DIR="logs"
DATASETS_DIR="datasets"


for data_type in non_nlp nlp; do
    for result_dir in $(ls -d ${ATTACKS_DIR}/${data_type}/*); do
        dataset=$(basename ${result_dir})
        for dir in $(ls -d ${result_dir}/*); do
            alg_name=$(basename ${dir})
            echo ">>>> Preparing data for ${dataset} dataset, ${alg_name} algorithm"
            PYTHONPATH=. python scripts/prepare_for_discr.py \
                --adversarial-dir ${dir} \
                --out-dir ${dir}/adv_detection


            echo ">>>> Training ${dataset} dataset, ${alg_name} algorithm"
            export DISCR_TRAIN_DATA_PATH=${dir}/adv_detection/train.json
            export DISCR_VALID_DATA_PATH=${dir}/adv_detection/test.json

            clf_dif=${LOGS_DIR}/${data_type}/dataset_${dataset}/target_clf/discriminator_${alg_name}
            allennlp train configs/models/classifier/gru_discriminator.jsonnet \
                -s ${clf_dif} \
                --force \
                --include-package adat

            echo ">>>> Evaluating Detector ${dataset} dataset, ${alg_name} algorithm"
            PYTHONPATH=. python scripts/calculate_discr_metrics.py \
                --adversarial-dir ${dir} \
                --classifier-dir ${clf_dif} \
                --test-path ${DISCR_VALID_DATA_PATH} \
                --cuda ${GPU_ID}
        done
    done
done


PYTHONPATH=. python scripts/aggregate_results.py \
    --results-dir ${LOGS_DIR} \
    --adv-detection
