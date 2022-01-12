##!/bin/bash
echo "Hi"
PERSONACHAT_SELF_URL=https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json
PERSONACHAT_SELF_ORIGIN=data/personachat_self_original.json

GPT_PERSONACHAT_URL=https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/gpt_personachat_cache.tar.gz
GPT_PERSONACHAT_CACHE=tmp/gpt_personachat_cache.tar.gz
GPT_PERSONACHAT_MODEL=model/gpt2_personachat

ENGAGING_CLASSIFIER_CACHE=tmp/engaging_classifier.tar.gz
ENGAGING_CLASSIFIER_MODEL=model/engaging_classifier

mkdir data
mkdir model
mkdir sample
mkdir tmp
mkdir $GPT_PERSONACHAT_MODEL
mkdir $ENGAGING_CLASSIFIER_MODEL

if [ ! -f $PERSONACHAT_SELF_ORIGIN ]; then
    echo "Downloading personachat dataset"
    wget $PERSONACHAT_SELF_URL -O $PERSONACHAT_SELF_ORIGIN
fi

if [ ! -f $GPT_PERSONACHAT_CACHE ]; then
    echo "Downloading GPT-2 model"
    wget $GPT_PERSONACHAT_URL -O $GPT_PERSONACHAT_CACHE
    echo "Extracting GPT-2 model"
    tar zxvf $GPT_PERSONACHAT_CACHE -C $GPT_PERSONACHAT_MODEL
fi

# TODO: upload engaging classifier model
if [ ! -f $ENGAGING_CLASSIFIER_MODEL ]; then
    echo "Can't find ${ENGAGING_CLASSIFIER_MODEL}! Extracting from ${ENGAGING_CLASSIFIER_CACHE}"
    tar zxvf $ENGAGING_CLASSIFIER_CACHE -C model
fi
