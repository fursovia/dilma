#!/usr/bin/env bash


# usage
# bash bin/evaluate_attacks.sh {ATTACKS_DIR} {MODEL_DIR} {GPU_ID}

ATTACKS_DIR=$1
MODEL_DIR=$2
GPU_ID=${3:-"-1"}
CLASSIFIER_NAME=$(basename ${MODEL_DIR})
METRICS_NAME=${CLASSIFIER_NAME}_metrics.json

for dir in $(ls -d ${ATTACKS_DIR}/*); do
    echo ">>>>>>>> Evaluating ${dir}"

    metrics_path=${dir}/${METRICS_NAME}
    attacked_data_path=${dir}/attacked_data.json
    if [[ ! -f "$metrics_path" ]] && [[ -f "$attacked_data_path" ]]; then
        PYTHONPATH=. python scripts/evaluate_attack.py \
            --adversarial-dir ${dir} \
            --classifier-dir ${MODEL_DIR} \
            --cuda ${GPU_ID}
    else
        echo "${metrics_path} is already calculated or ${attacked_data_path} does not exist"
    fi
done