# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
import warnings

from argparse import ArgumentParser
from itertools import chain

from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from torch.nn.utils.rnn import pad_sequence
from train import get_data_loaders

from tensorboardX import SummaryWriter

from Chatbot import prepare_chatbot, get_score

device_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter("runs")
SPECIAL_TOKENS = ["<bos>", "<|eos|>", "<speaker1>", "<speaker2>", "<pad>"]

temperature = 1


def generate_response(history, tokenizer, model, args, current_output=None):
    """
    Generate response without persona.
    """
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    bos, eos, spk1, spk2, pad = special_tokens_ids
    if current_output is None:
        current_output = [[] for _ in range(arg.train_batch_size)]

    sequence_bt = [[[bos]] + history_i for history_i in history]
    sequence_bt = [
        [seq[0]]
        + [[spk2 if (len(seq) - i) % 2 else spk1] + s for i, s in enumerate(seq[1:])]
        for seq in sequence_bt
    ]
    token_type_ids_bt = [
        [spk2 if i % 2 else spk1 for i, s in enumerate(seq) for _ in s]
        for seq in sequence_bt
    ]
    sequence_bt = [list(chain(*seq)) for seq in sequence_bt]
    mask_len = [len(x) for x in sequence_bt]
    mass = []
    for i in range(len(sequence_bt)):
        m = [1 for j in range(mask_len[i])]
        mass.append(m[:])

    sequence_bt = pad_sequence(
        [torch.LongTensor(x) for x in sequence_bt],
        batch_first=True,
        padding_value=tokenizer.encode("<pad>")[0],
    ).to(args.device)
    token_type_ids_bt = pad_sequence(
        [torch.LongTensor(x) for x in token_type_ids_bt],
        batch_first=True,
        padding_value=spk1,
    ).to(args.device)
    mask = pad_sequence(
        [torch.LongTensor(x) for x in mass], batch_first=True, padding_value=0
    ).to(args.device)

    _, past = model(sequence_bt, attention_mask=mask, token_type_ids=token_type_ids_bt)
    token_tp = torch.LongTensor([[spk2] if len(x) % 2 else [spk1] for x in history]).to(
        args.device
    )
    prev = torch.LongTensor([[spk2] if len(x) % 2 else [spk1] for x in history]).to(
        args.device
    )

    temp_sen = [[] for i in range(len(history))]
    for i_word in range(args.max_length):

        logits, past = model(prev, token_type_ids=token_tp, past=past)
        logits = logits.squeeze(0).squeeze(1)
        # logits = top_filtering(logits, top_k=arg.top_k, top_p=arg.top_p)
        probs = torch.softmax(logits, dim=-1)

        prev = torch.multinomial(probs, num_samples=1)
        # prev = [torch.topk(prob_i, 1)[1] if arg.no_sample else torch.multinomial(prob_i, 1) for prob_i in probs]
        for prev_i, prob_i in zip(prev, probs):
            if i_word < args.min_length and prev_i.item() in special_tokens_ids:
                while prev_i.item() in special_tokens_ids:
                    if prob_i.max().item() == 1:
                        warnings.warn(
                            "Warning: model generating special token with probability 1."
                        )
                        break  # avoid infinitely looping over special token
                    prev_i = torch.multinomial(prob_i, num_samples=1)

        if i_word == 0:
            for j in range(len(history)):
                temp_sen[j].append(prev[j].item())
            continue
        flag = 1

        for j in range(0, len(history)):
            if temp_sen[j][-1] not in special_tokens_ids:
                flag = 0
                temp_sen[j].append(prev[j].item())
        if flag == 1:
            break

    for i in range(len(temp_sen)):
        if temp_sen[i][-1] == eos:
            temp_sen[i][:] = temp_sen[i][:-1]
    return temp_sen


def train(chatbot, interlocutor, tokenizer, train_loader, args, args_bot):
    optimizer = AdamW(chatbot.parameters(), lr=1e-5, eps=1e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=500, num_training_steps=8000
    )

    selector_history = 5

    chatbot.train()
    optimizer.zero_grad()
    for i_epoch in range(args.epoch):
        i_batch = 0
        sc = 0
        prob_record = []
        score_record = []
        total_loss = 0
        for input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids in tqdm(
            train_loader
        ):
            # ===== set up history ===============
            # interlocutor_persona_enc = []
            history_enc = []
            for input_i in input_ids:
                history = []
                sen = []
                per = True
                sen_spk2 = True
                for i in range(1, len(input_i[0])):
                    if input_i[0][i] != 50258:  # <|eos|>
                        if per and input_i[0][i] == 50261:  # <speaker2>
                            per = False
                        else:
                            if sen_spk2:
                                if input_i[0][i] == 50260:
                                    sen_spk2 = False
                                    history.append(sen)
                                    sen = []
                                else:
                                    sen.append(int(input_i[0][i]))
                            else:
                                if input_i[0][i] == 50261:
                                    sen_spk2 = True
                                    history.append(sen)
                                    sen = []
                                else:
                                    sen.append(int(input_i[0][i]))
                    else:
                        history.append(sen)
                        break
                history_enc.append(history)
            print("######################")
            print("history init")
            for sen_enc in history_enc[0]:
                print(tokenizer.decode(sen_enc))

            # ===== generate s1 from chatbot ==========
            with torch.no_grad():
                response_enc = generate_response(
                    history_enc, tokenizer, chatbot, args_bot
                )
                for i in range(args.batch_size):
                    history_enc[i].append(response_enc[i])

            score_bot = get_score([h[-2:] for h in history_enc], tokenizer)

            # ===== generate s2 from interlocutor ==========
            with torch.no_grad():
                response_enc = generate_response(
                    history_enc, tokenizer, interlocutor, args_bot
                )
                for i in range(args.batch_size):
                    history_enc[i].append(response_enc[i])

            score = get_score([h[-2:] for h in history_enc], tokenizer)

            loss = 0
            for sb, s in zip(score_bot, score):
                loss -= s + 0.2 * sb
            loss.backward()

            if i_batch % 2 == 0:
                torch.nn.utils.clip_grad_norm_(chatbot.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if i_batch % args.log_step == 0:
                niter = i_epoch * len(train_loader) + i_batch
                writer.add_scalar("Train/Loss", loss, niter)
                writer.add_scalar("Train/score_bot", score_bot, niter)
                writer.add_scalar("Train/score", score, niter)
            if i_batch % args.save_time_step == 0:
                chatbot.save_pretrained(
                    os.path.join(args.work_space, args.save_dir, args.model_name),
                )


def main():
    parser = ArgumentParser()
    # path
    parser.add_argument("--work_space", type=str, default=".")
    parser.add_argument("--model_checkpoint", type=str, default="model/gpt2/")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save_dir", type=str, default="model/")
    parser.add_argument(
        "--model_name", type=str, default="one"
    )  # persona selector model folder

    # steps
    parser.add_argument("--log_step", type=int, default=2)
    parser.add_argument("--print_sample_step", type=int, default=20)
    parser.add_argument("--save_time_step", type=int, default=100)

    args = parser.parse_args()
    os.makedirs(
        os.path.join(args.work_space, args.save_dir, args.model_name), exist_ok=True
    )

    # ===== prepare dataset, models and optimizer ==========
    chatbot, interlocutor, tokenizer, args_bot = prepare_chatbot(
        "gpt2", bt=args.batch_size
    )
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(
        args_bot, tokenizer
    )
    del val_loader, train_sampler, valid_sampler
    print("\n\nlen(train_loader): ", len(train_loader), "\n\n")

    train(chatbot, interlocutor, tokenizer, train_loader, args, args_bot)


if __name__ == "__main__":
    main()
