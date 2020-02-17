#!/usr/bin/env bash

# usage
# sh bin/run_attacks.sh {DATA_DIR} {GPU_ID}
# {COPYNET_DIR} {CLASSIFIER_COPYNET_DIR} {DEEP_LEVENSHTEIN_COPYNET_DIR}
# {CLASSIFIER_BASIC_DIR}
# {BEAM_SIZE} {SAMPLE}

DATA_DIR=$1
BASENAME=$(basename ${DATA_DIR})
GPU_ID=${2:-0}
COPYNET_DIR=$3
CLASSIFIER_COPYNET_DIR=$4
DEEP_LEVENSHTEIN_COPYNET_DIR=$5
CLASSIFIER_BASIC_DIR=$6
BEAM_SIZE=${7:-3}
SAMPLE=${8:-1000}

CASCADA_DIR=results/${BASENAME}/cascada
MCMC_DIR=results/${BASENAME}/mcmc
RANDOM_DIR=results/${BASENAME}/random
HOTFLIP_DIR=results/${BASENAME}/hotflip


echo "Running MCMC..."
python run_mcmc.py \
    --csv_path ${DATA_DIR}/test.csv \
    --results_path ${MCMC_DIR} \
    --copynet_path ${COPYNET_DIR} \
    --classifier_path ${CLASSIFIER_BASIC_DIR} \
    --beam_size ${BEAM_SIZE} \
    --sample ${SAMPLE} \
    --cuda ${GPU_ID}


echo "Running Random sampler..."
python run_mcmc.py \
    --csv_path ${DATA_DIR}/test.csv \
    --results_path ${RANDOM_DIR} \
    --copynet_path ${COPYNET_DIR} \
    --classifier_path ${CLASSIFIER_BASIC_DIR} \
    --random \
    --beam_size ${BEAM_SIZE} \
    --sample ${SAMPLE} \
    --cuda ${GPU_ID}


echo "Running HotFlip..."
python run_hotflip.py \
    --csv_path ${DATA_DIR}/test.csv \
    --results_path ${HOTFLIP_DIR} \
    --classifier_path ${CLASSIFIER_BASIC_DIR} \
    --sample ${SAMPLE}


echo "Running CASCADA..."
python run_cascada.py \
    --csv_path ${DATA_DIR}/test.csv \
    --results_path ${CASCADA_DIR} \
    --copynet_path ${COPYNET_DIR} \
    --classifier_path ${CLASSIFIER_COPYNET_DIR} \
    --levenshtein_path ${DEEP_LEVENSHTEIN_COPYNET_DIR} \
    --beam_size ${BEAM_SIZE} \
    --sample ${SAMPLE} \
    --cuda ${GPU_ID}
