#!/usr/bin/env bash

# usage
# bash bin/evaluate.sh {RESULTS_DIR} {LOG_DIR} {GPU_ID}

RESULTS_DIR=${1:-"results"}
LOG_DIR=${2:-"logs"}
default_gpu_id=0
GPU_ID=${3:-$default_gpu_id}


for data_type in nlp non_nlp; do
    for result_dir in $(ls -d ${RESULTS_DIR}/${data_type}/*); do
        dataset=$(basename ${result_dir})
        target_clf_dir=${LOG_DIR}/${data_type}/${dataset}/target_clf

        for dir in $(ls -d ${result_dir}/*); do
            PYTHONPATH=. python scripts/evaluate_attack.py \
                --adversarial-dir ${dir} \
                --classifier-dir ${target_clf_dir} \
                --cuda ${GPU_ID}
        done
    done
done
