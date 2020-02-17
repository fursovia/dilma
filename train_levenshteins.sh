#!/usr/bin/env bash

#for dataset in ai_academy_data ag_news kaggle_transactions insurance
for dataset in ai_academy_data kaggle_transactions insurance
do
    echo "START: ${dataset}"
    python train.py --task deep_levenshtein_copynet --model_dir experiments/${dataset}/deep_levenshtein_copynet --data_dir data/${dataset}/levenshtein --copynet_dir experiments/${dataset}/nonmasked_copynet_with_attention --cuda 0 --batch_size 256
done