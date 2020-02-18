#!/usr/bin/env bash

#for dataset in ai_academy_data ag_news kaggle_transactions insurance
for dataset in insurance
do
    echo "START: ${dataset}"
    python train_eval_model.py -dd data/${dataset} -mp results/${dataset}/logit_tfidf.model -mqp results/${dataset}/logit_tfidf_metric.json
done