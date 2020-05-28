#!/usr/bin/env bash

ATTACKS_DIR=attacks



for num in 50 100 500 1000 5000; do
    for name in ai_academy insurance transactions ag sst trec mr; do
        for dir in $(ls -d ${ATTACKS_DIR}/${name}/*); do

            PYTHONPATH=. python scripts/prepare_for_fine_tuning.py \
                --adversarial-dir ${dir} \
                --mix-with-path datasets/${name}/original_class/train.json \
                --num-examples ${num}
        done
    done
done