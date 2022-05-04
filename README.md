# Persona-Chatbot-Project
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## Prepare:  
download folling files  
https://drive.google.com/file/d/1m2jk6NdhTEHVzQn3GTAtE3-CKYv1m658/view?usp=sharing  
https://drive.google.com/file/d/1Yijsy8Zot4icvkaQSkpRJ5hj_3NnWDK4/view?usp=sharing  



```bash
tar zxvf model.tar.gz  
tar zxvf data.tar.gz  
# then move model and data folder into this repo folder
```

have a pre-train persona bot
```bash
git clone https://github.com/huggingface/transfer-learning-conv-ai  
cd transfer-learning-conv-ai
python -m torch.distributed.launch --nproc_per_node=1 ./train.py --gradient_accumulation_steps=4 --lm_coef=2.0 --max_history=2 --n_epochs=1 --num_candidates=4 --personality_permutations=2 --train_batch_size=8 --valid_batch_size=8  
```

## Usage:
    python3 Chatbot.py  

## TODO:  


## Reference:
https://github.com/huggingface/transfer-learning-conv-ai