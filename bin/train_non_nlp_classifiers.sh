#!/usr/bin/env bash


export LM_VOCAB_PATH=logs/nlp_lm/vocabulary


LOG_DIR=new_logs/ag
DATA_DIR=datasets/ag
export CLS_NUM_CLASSES=4

echo ">>>>>>>>> AG"
echo ">>>>>>> Training Target Classifier"
export CLS_TRAIN_DATA_PATH=${DATA_DIR}/original_class/train.json
export CLS_VALID_DATA_PATH=${DATA_DIR}/original_class/valid.json

allennlp train configs/classifier/gru_classifier.jsonnet \
    -s ${LOG_DIR}/original_class_gru \
    --include-package adat


echo ">>>>>>> Training Substitute Classifier"
export CLS_TRAIN_DATA_PATH=${DATA_DIR}/substitute_class/train.json
export CLS_VALID_DATA_PATH=${DATA_DIR}/substitute_class/valid.json

allennlp train configs/classifier/gru_classifier.jsonnet \
    -s ${LOG_DIR}/substitute_class_gru \
    --include-package adat


LOG_DIR=new_logs/trec
DATA_DIR=datasets/trec
export CLS_NUM_CLASSES=6

echo ">>>>>>>>> TREC"
echo ">>>>>>> Training Target Classifier"
export CLS_TRAIN_DATA_PATH=${DATA_DIR}/original_class/train.json
export CLS_VALID_DATA_PATH=${DATA_DIR}/original_class/valid.json

allennlp train configs/classifier/gru_classifier.jsonnet \
    -s ${LOG_DIR}/original_class_gru \
    --include-package adat


echo ">>>>>>> Training Substitute Classifier"
export CLS_TRAIN_DATA_PATH=${DATA_DIR}/substitute_class/train.json
export CLS_VALID_DATA_PATH=${DATA_DIR}/substitute_class/valid.json

allennlp train configs/classifier/gru_classifier.jsonnet \
    -s ${LOG_DIR}/substitute_class_gru \
    --include-package adat

#
LOG_DIR=new_logs/sst
DATA_DIR=datasets/sst
export CLS_NUM_CLASSES=2

echo ">>>>>>>>> SST"
echo ">>>>>>> Training Target Classifier"
export CLS_TRAIN_DATA_PATH=${DATA_DIR}/original_class/train.json
export CLS_VALID_DATA_PATH=${DATA_DIR}/original_class/valid.json

allennlp train configs/classifier/gru_classifier.jsonnet \
    -s ${LOG_DIR}/original_class_gru \
    --include-package adat


echo ">>>>>>> Training Substitute Classifier"
export CLS_TRAIN_DATA_PATH=${DATA_DIR}/substitute_class/train.json
export CLS_VALID_DATA_PATH=${DATA_DIR}/substitute_class/valid.json

allennlp train configs/classifier/gru_classifier.jsonnet \
    -s ${LOG_DIR}/substitute_class_gru \
    --include-package adat
