##!/bin/bash
echo "Hi"
mkdir data
mkdir model
mkdir sample
mkdir tmp

ENGAGING_CLASSIFIER_MODEL=./model/engaging_classifier/

wget "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json" -O data/personachat_self_original.json
wget "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/gpt_personachat_cache.tar.gz" -O tmp/gpt_personachat_cache.tar.gz
mkdir model/gpt2_personachat
tar zxvf tmp/gpt_personachat_cache.tar.gz -C model/gpt2_personachat

if [ ! -f $ENGAGING_CLASSIFIER_MODEL ]; then
    echo "The folder ${ENGAGING_CLASSIFIER_MODEL} does not exist! Extracting the model from tmp/engaging_classifier.tar.gz"
    tar zxvf tmp/engaging_classifier.tar.gz -C model/engaging_classifier
fi
