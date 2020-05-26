#!/usr/bin/env bash


LOG_DIR=new_logs/insurance
DATA_DIR=datasets/insurance
export LM_VOCAB_PATH=logs/insurance/lm/vocabulary
export CLS_NUM_CLASSES=2

echo ">>>>>>>>> Insurance"
echo ">>>>>>> Training Target Classifier"
export CLS_TRAIN_DATA_PATH=${DATA_DIR}/original_class/train.json
export CLS_VALID_DATA_PATH=${DATA_DIR}/original_class/valid.json

allennlp train configs/classifier/cnn_classifier.jsonnet \
    -s ${LOG_DIR}/original_class_cnn \
    --include-package adat


echo ">>>>>>> Training Substitute Classifier"
export CLS_TRAIN_DATA_PATH=${DATA_DIR}/substitute_class/train.json
export CLS_VALID_DATA_PATH=${DATA_DIR}/substitute_class/valid.json

allennlp train configs/classifier/cnn_classifier.jsonnet \
    -s ${LOG_DIR}/substitute_class_cnn \
    --include-package adat


LOG_DIR=new_logs/ai_academy
DATA_DIR=datasets/ai_academy
export LM_VOCAB_PATH=logs/ai_academy/lm/vocabulary
export CLS_NUM_CLASSES=4

echo ">>>>>>>>> AGE"
echo ">>>>>>> Training Target Classifier"
export CLS_TRAIN_DATA_PATH=${DATA_DIR}/original_class/train.json
export CLS_VALID_DATA_PATH=${DATA_DIR}/original_class/valid.json

allennlp train configs/classifier/cnn_classifier.jsonnet \
    -s ${LOG_DIR}/original_class_cnn \
    --include-package adat


echo ">>>>>>> Training Substitute Classifier"
export CLS_TRAIN_DATA_PATH=${DATA_DIR}/substitute_class/train.json
export CLS_VALID_DATA_PATH=${DATA_DIR}/substitute_class/valid.json

allennlp train configs/classifier/cnn_classifier.jsonnet \
    -s ${LOG_DIR}/substitute_class_cnn \
    --include-package adat

LOG_DIR=new_logs/transactions
DATA_DIR=datasets/transactions
export LM_VOCAB_PATH=logs/transactions/lm/vocabulary
export CLS_NUM_CLASSES=2

echo ">>>>>>>>> SST"
echo ">>>>>>> Training Target Classifier"
export CLS_TRAIN_DATA_PATH=${DATA_DIR}/original_class/train.json
export CLS_VALID_DATA_PATH=${DATA_DIR}/original_class/valid.json

allennlp train configs/classifier/cnn_classifier.jsonnet \
    -s ${LOG_DIR}/original_class_cnn \
    --include-package adat


echo ">>>>>>> Training Substitute Classifier"
export CLS_TRAIN_DATA_PATH=${DATA_DIR}/substitute_class/train.json
export CLS_VALID_DATA_PATH=${DATA_DIR}/substitute_class/valid.json

allennlp train configs/classifier/cnn_classifier.jsonnet \
    -s ${LOG_DIR}/substitute_class_cnn \
    --include-package adat
