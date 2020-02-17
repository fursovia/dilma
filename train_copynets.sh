#!/usr/bin/env bash

#for dataset in ai_academy_data ag_news kaggle_transactions insurance
for dataset in ai_academy_data kaggle_transactions insurance
do
    echo "START: ${dataset}"
    python train.py --task nonmasked_copynet_with_attention --model_dir experiments/${dataset}/nonmasked_copynet_with_attention --data_dir data/${dataset} --use_mask --cuda 1 --batch_size 256
    python train.py --task nonmasked_copynet --model_dir experiments/${dataset}/nonmasked_copynet --data_dir data/${dataset} --cuda 1  --batch_size 256
done