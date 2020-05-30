#!/usr/bin/env bash

# ATTACKS_DIR=non_nlp_attacks_5000
# ATTACKS_DIR=valid_attacks


for num in 5000 50 100 500 1000; do
    for name in ai_academy insurance transactions; do
        # dir=${ATTACKS_DIR}/${name}/hotflip_cnn
        for dir in $(ls -d ${ATTACKS_DIR}/${name}/*); do

            PYTHONPATH=. python scripts/prepare_for_fine_tuning.py \
                --adversarial-dir ${dir} \
                --mix-with-path datasets/${name}/original_class/train.json \
                --num-examples ${num}
        done
    done
done