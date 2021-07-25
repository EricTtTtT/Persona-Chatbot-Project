# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
import warnings

from argparse import ArgumentParser
from itertools import chain, permutations

from tqdm.auto import tqdm
import numpy as np
import torch

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from train import get_data_loaders_DialoGPT

from tensorboardX import SummaryWriter

from Chatbot import get_score

# device_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device_1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter("runs")
# SPECIAL_TOKENS = ["<bos>", "<|eos|>", "<speaker1>", "<speaker2>", "<pad>"]

temperature = 1


def generate_response(input_ids, mask, tokenizer, model, args):
    """
    Generate response without persona.
    """
    eos_id = tokenizer.eos_token_id
    bt = args.train_batch_size

    input_ids = input_ids.to(args.device)
    mask = mask.to(args.device)
    _, past = model(input_ids, attention_mask=mask)
    prev = torch.LongTensor([[tokenizer.eos_token_id] for _ in range(bt)]).to(
        args.device
    )
    mask_append = torch.tensor([[1] for i in range(bt)]).to(args.device)
    mask = torch.cat((mask, mask_append), 1)
    temp_sen = [[] for i in range(bt)]

    os.system("nvidia-smi")
    for i_word in range(args.max_length):
        logits, past = model(prev, past=past, attention_mask=mask)
        mask = torch.cat((mask, mask_append), 1)
        logits = logits.squeeze(0).squeeze(1)
        probs = torch.softmax(logits, dim=-1)

        prev = torch.multinomial(probs, num_samples=1)

        if i_word == 0:
            for j in range(bt):
                temp_sen[j].append(prev[j].item())
            continue

        flag = 1
        for j in range(0, bt):
            if temp_sen[j][-1] != eos_id:
                flag = 0
                temp_sen[j].append(prev[j].item())
        if flag == 1:
            break
    os.system("nvidia-smi")

    for i in range(bt):
        if temp_sen[i][-1] == eos_id:
            temp_sen[i][:] = temp_sen[i][:-1]
    return temp_sen


def train(chatbot, interlocutor, tokenizer, train_loader, args, args_bot):
    optimizer = AdamW(chatbot.parameters(), lr=1e-5, eps=1e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=500, num_training_steps=8000
    )

    optimizer.zero_grad()
    for i_epoch in range(args.epoch):
        for batch in tqdm(train_loader):
            input_ids, attention_mask = batch
            chatbot_ret = generate_response(
                input_ids, attention_mask, tokenizer, chatbot, args_bot
            )
            for input, reply in zip(input_ids, chatbot_ret):
                print("\n#################")
                print(tokenizer.decode(input))
                print(tokenizer.decode(reply))
            exit()

            # outputs = chatbot.generate(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     pad_token_id=0,
            #     max_length=200,
            #     do_sample=True,
            # )


class ARG:
    def __init__(self, bt=8):
        self.dataset_path = "data/personachat_self_original.json"
        self.dataset_cache = "data/dataset_cache"
        self.max_history = 2
        self.num_candidates = 1
        self.device = "cuda"
        self.no_sample = False
        self.max_length = 20
        self.min_length = 1
        self.seed = 2
        self.temperature = 1
        self.top_k = 0
        self.top_p = 0
        self.distributed = False
        self.local_rank = -1
        self.train_batch_size = bt
        self.valid_batch_size = bt


def main():
    parser = ArgumentParser()
    # path
    parser.add_argument("--work_space", type=str, default=".")
    parser.add_argument(
        "--model_checkpoint", type=str, default="model/dialogpt-medium/"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save_dir", type=str, default="model/")
    parser.add_argument("--model_name", type=str, default="dialo_one")

    # steps
    parser.add_argument("--log_step", type=int, default=2)
    parser.add_argument("--print_sample_step", type=int, default=20)
    parser.add_argument("--save_time_step", type=int, default=100)

    args = parser.parse_args()
    args_bot = ARG(args.batch_size)

    os.makedirs(
        os.path.join(args.work_space, args.save_dir, args.model_name), exist_ok=True
    )

    # ===== prepare dataset, models and optimizer ==========
    # "microsoft/DialoGPT-medium"
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    # chatbot = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    chatbot = AutoModelForCausalLM.from_pretrained(args.model_checkpoint)
    interlocutor = AutoModelForCausalLM.from_pretrained(args.model_checkpoint)

    chatbot.to(args_bot.device).train()
    # interlocutor.to(args_bot.device).eval()
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token = tokenizer.decode([0])

    # history = [
    #     "good evening, how are you? <|endoftext|> coupons are awesome. how are you? <|endoftext|> i love coupon cutting. i detest school. <|endoftext|> you should coupon with me. saves a ton of money! <|endoftext|>"
    # ]
    # # length of history is 45
    # h_enc = tokenizer(history, max_length=45, padding="max_length", return_tensors="pt")
    # # h_enc = tokenizer(history, max_length=50, padding='max_length', return_tensors='pt')
    # print(h_enc)
    # input_ids = h_enc["input_ids"].to(args_bot.device)
    # attention_mask = h_enc["attention_mask"].to(args_bot.device)
    # outputs = chatbot.generate(
    #     input_ids=input_ids,
    #     attention_mask=attention_mask,
    #     pad_token_id=0,
    #     max_length=100,
    #     do_sample=True,
    # )
    # print(tokenizer.batch_decode(outputs))
    # exit()

    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders_DialoGPT(
        args_bot, tokenizer
    )

    del val_loader, train_sampler, valid_sampler

    train(chatbot, interlocutor, tokenizer, train_loader, args, args_bot)


if __name__ == "__main__":
    main()
