##!/bin/bash
echo "Hi"
mkdir data
mkdir model
mkdir sample
mkdir tmp

wget "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json" -O data/personachat_self_original.json
wget "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/gpt_personachat_cache.tar.gz" -O tmp/gpt_personachat_cache.tar.gz
mkdir model/gpt2-personachat
tar zxvf tmp/gpt_personachat_cache.tar.gz -C model/gpt2-personachat
