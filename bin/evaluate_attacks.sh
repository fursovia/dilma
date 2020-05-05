#!/usr/bin/env bash


# usage
# bash bin/evaluate_attacks.sh {ATTACKS_DIR} {MODEL_DIR}

ATTACKS_DIR=$1
MODEL_DIR=$2

for dir in $(ls -d ${ATTACKS_DIR}/*); do
    echo ">>>>>>>> Evaluating ${dir}"
    PYTHONPATH=. python scripts/evaluate_attack.py \
        --adversarial-dir ${dir} \
        --classifier-dir ${MODEL_DIR}
done