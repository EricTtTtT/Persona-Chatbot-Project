# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import math
import random
import logging
import warnings

from argparse import ArgumentParser
from itertools import chain, permutations

from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from train import get_data_loaders_DialoGPT

from tensorboardX import SummaryWriter

from Chatbot import get_score
import gc

writer = SummaryWriter("runs")

def generate_response(input_ids, mask, tokenizer, model, args, model_2=None, device="cuda:0"):
    """
    Generate response without persona.
    """
    eos_id = tokenizer.eos_token_id
    bt = args.train_batch_size

    with torch.no_grad():
        input_ids = input_ids.to(device)
        mask = mask.to(device)
        _, past = model(input_ids, past=None, attention_mask=mask)
        if model_2:
            input_ids = input_ids.to(args.device_2)
            mask = mask.to(args.device_2)
            _, past_2 = model_2(input_ids, past=None, attention_mask=mask)
            input_ids = input_ids.to(args.device)
            mask = mask.to(device)

<<<<<<< HEAD
        prev = torch.LongTensor([[tokenizer.eos_token_id] for _ in range(bt)]).to(device)
=======
        prev = torch.LongTensor([[tokenizer.eos_token_id] for _ in range(bt)]).to(
            device
        )
>>>>>>> 4c9672d0847152fb808b125e45cfde77cd3a863a
        mask_append = torch.tensor([[1] for i in range(bt)]).to(device)
        temp_sen = [[] for _ in range(bt)]
        log_prob = [[] for _ in range(bt)]
        coherence_score_arr = [[] for _ in range(bt)]
        coherence_loss = [0 for _ in range(bt)]
        cont = [True for _ in range(bt)]

        for i_word in range(args.max_length):
            mask = torch.cat((mask, mask_append), 1)
            logits, past = model(prev, past=past, attention_mask=mask)
            logits = logits.squeeze(0).squeeze(1)
            logits = torch.softmax(logits, dim=-1)

            if model_2:
                mask = mask.to(args.device_2)
                prev = prev.to(args.device_2)
                logits_2, past_2 = model_2(prev, past=past_2, attention_mask=mask)
                logits_2 = torch.softmax(logits_2.squeeze(0).squeeze(1), dim=-1)
                mask = mask.to(device)
                prev = prev.to(device)
<<<<<<< HEAD
                

            prev = torch.multinomial(logits, num_samples=1)
            log_p = [math.log(p[idx]) for idx, p in zip(prev, logits)]
            
=======

            prev = torch.multinomial(logits, num_samples=1)
            log_p = [math.log(p[idx]) for idx, p in zip(prev, logits)]

>>>>>>> 4c9672d0847152fb808b125e45cfde77cd3a863a
            if model_2:
                # get average of probs_2
                probs_2 = []
                for j in range(bt):
                    if cont[j]:
                        probs_2.append(logits_2[j][prev[j].item()].item())
                if len(probs_2) == 0:
                    avg_prob_2 = 0
                else:
                    avg_prob_2 = sum(probs_2) / len(probs_2)
<<<<<<< HEAD
            
=======

>>>>>>> 4c9672d0847152fb808b125e45cfde77cd3a863a
            if i_word == 0:
                for j in range(bt):
                    # print("\n##########")
                    # print(prev[j].item(), logits[j][prev[j].item()].item(), logits_2[j][prev[j].item()].item())
                    temp_sen[j].append(prev[j].item())
                    log_prob[j].append(log_p[j])
                    if model_2:
<<<<<<< HEAD
                        tmp_loss = F.cross_entropy(logits[j].unsqueeze(0), prev.view(-1)[j].unsqueeze(0))
                        coherence_score_arr[j].append(logits_2[j][prev[j].item()].item())
                        coherence_loss[j] += (logits_2[j][prev[j].item()].item() - avg_prob_2) * tmp_loss
                continue
            
=======
                        tmp_loss = F.cross_entropy(
                            logits[j].unsqueeze(0), prev.view(-1)[j].unsqueeze(0)
                        )
                        coherence_score_arr[j].append(
                            logits_2[j][prev[j].item()].item()
                        )
                        coherence_loss[j] += (
                            logits_2[j][prev[j].item()].item() - avg_prob_2
                        ) * tmp_loss
                continue

>>>>>>> 4c9672d0847152fb808b125e45cfde77cd3a863a
            for j in range(bt):
                if cont[j]:
                    if temp_sen[j][-1] != eos_id:
                        temp_sen[j].append(prev[j].item())
                        log_prob[j].append(log_p[j])
                        if model_2:
<<<<<<< HEAD
                            tmp_loss = F.cross_entropy(logits[j].unsqueeze(0), prev.view(-1)[j].unsqueeze(0))
                            coherence_score_arr[j].append(logits_2[j][prev[j].item()].item())
                            coherence_loss[j] += (logits_2[j][prev[j].item()].item() - avg_prob_2) * tmp_loss
=======
                            tmp_loss = F.cross_entropy(
                                logits[j].unsqueeze(0), prev.view(-1)[j].unsqueeze(0)
                            )
                            coherence_score_arr[j].append(
                                logits_2[j][prev[j].item()].item()
                            )
                            coherence_loss[j] += (
                                logits_2[j][prev[j].item()].item() - avg_prob_2
                            ) * tmp_loss
>>>>>>> 4c9672d0847152fb808b125e45cfde77cd3a863a
                    else:
                        cont[j] = False
            if not (True in cont):
                break
<<<<<<< HEAD
        
=======

>>>>>>> 4c9672d0847152fb808b125e45cfde77cd3a863a
        for i in range(bt):
            if temp_sen[i][-1] == eos_id:
                temp_sen[i][:] = temp_sen[i][:-1]
                log_prob[i][:] = log_prob[i][:-1]
<<<<<<< HEAD
        
=======

>>>>>>> 4c9672d0847152fb808b125e45cfde77cd3a863a
        coherence_score = [0 for _ in range(bt)]
        if model_2:
            for j, scores in enumerate(coherence_score_arr):
                if len(scores) != 0:
                    coherence_score[j] = sum(scores) / len(scores)

        return temp_sen, log_prob, coherence_score, coherence_loss


def train(chatbot, interlocutor, tokenizer, train_loader, args, args_bot):
    chatbot.train()
    interlocutor.eval()

    optimizer = AdamW(chatbot.parameters(), lr=1e-5, eps=1e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=2000, num_training_steps=8000
    )

    optimizer.zero_grad()
    i_iter = 0
    for i_epoch in range(args.epoch):
        i_batch = 0
        running_loss = 0
        running_score = 0
        running_coherence = 0

        for batch in tqdm(train_loader):
            input_ids, mask = batch
<<<<<<< HEAD
            chatbot_reply, log_prob, coherence_score, coherence_loss = generate_response(
                input_ids, mask, tokenizer, chatbot, args_bot,
                model_2=interlocutor, device=args_bot.device
=======
            (
                chatbot_reply,
                log_prob,
                coherence_score,
                coherence_loss,
            ) = generate_response(
                input_ids,
                mask,
                tokenizer,
                chatbot,
                args_bot,
                model_2=interlocutor,
                device=args_bot.device,
>>>>>>> 4c9672d0847152fb808b125e45cfde77cd3a863a
            )

            # padding reply
            max_l = max((len(reply) for reply in chatbot_reply))
            reply_tensor = []
            for reply in chatbot_reply:
                reply_tensor.append(
                    [tokenizer.eos_token_id]
                    + reply
                    + [tokenizer.pad_token_id for _ in range(max_l - len(reply))]
                )
            reply_tensor = torch.tensor(reply_tensor)
            input_ids = torch.cat((input_ids, reply_tensor), 1)
<<<<<<< HEAD
            mask_append = [[1] * len(reply) + [0] * (max_l - len(reply)) + [1] for reply in chatbot_reply]
            mask_append = torch.tensor(mask_append)
            mask = torch.cat((mask, mask_append), 1)

            spk2_reply, _, coherence_score_spk2, coherence_loss_spk2 = generate_response(
                input_ids, mask, tokenizer, interlocutor, args_bot,
                device=args_bot.device_2
=======
            mask_append = torch.ones((args.batch_size, max_l + 1))  # +1 for eos_token
            mask = torch.cat((mask, mask_append), 1)

            (
                spk2_reply,
                _,
                coherence_score_spk2,
                coherence_loss_spk2,
            ) = generate_response(
                input_ids,
                mask,
                tokenizer,
                interlocutor,
                args_bot,
                device=args_bot.device_2,
>>>>>>> 4c9672d0847152fb808b125e45cfde77cd3a863a
            )

            # get engagin score
            q_r = [[q, r] for q, r in zip(chatbot_reply, spk2_reply)]
            score = get_score(q_r, tokenizer)
            score_mean = sum(score) / len(score)
            running_score += score_mean

            running_coherence += sum(coherence_score) / len(coherence_score)

            # calculate reward
            rewards = [sc - score_mean for sc in score]
            # rewards = [sc - score_mean + coh_sc*args.weight_coherence for sc, coh_sc in zip(score, coherence_score)]
            for r, log_p, coh_loss in zip(rewards, log_prob, coherence_loss):
                if len(log_p) == 0:
                    logging.info("Error! divided by 0 due to empty log_p")
                    continue
                running_loss -= r * sum(log_p) / len(log_p)
                running_loss += coh_loss * args.weight_coherence

            i_batch += 1
            if i_batch % args.step_optimize == 0:
                # loss = torch.tensor(
                #     running_loss / args.step_optimize, requires_grad=True
                # )
<<<<<<< HEAD
                loss = running_loss.clone().detach().requires_grad_(True) / args.step_optimize
                loss.backward()

                torch.nn.utils.clip_grad_norm_(chatbot.parameters(), .5)
=======
                loss = (
                    running_loss.clone().detach().requires_grad_(True)
                    / args.step_optimize
                )
                loss.backward()

                torch.nn.utils.clip_grad_norm_(chatbot.parameters(), 0.5)
>>>>>>> 4c9672d0847152fb808b125e45cfde77cd3a863a
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                writer.add_scalar("Train/Loss", loss, i_iter)
                writer.add_scalar(
                    "Train/Engaging_score", running_score / args.step_optimize, i_iter
                )
                writer.add_scalar(
<<<<<<< HEAD
                    "Train/Coherence_score", running_coherence / args.step_optimize, i_iter
=======
                    "Train/Coherence_score",
                    running_coherence / args.step_optimize,
                    i_iter,
>>>>>>> 4c9672d0847152fb808b125e45cfde77cd3a863a
                )
                i_iter += 1

                running_loss = 0
                running_score = 0
                running_coherence = 0

            if i_batch % args.step_sample == 0:
                input_ids_decoded = tokenizer.batch_decode(input_ids)
                spk2_reply_decoded = tokenizer.batch_decode(spk2_reply)
                for dialogue, spk2 in zip(input_ids_decoded, spk2_reply_decoded):
                    print("\n#########################")
                    print(
                        dialogue.replace(tokenizer.eos_token, "\n").replace(
                            tokenizer.pad_token, ""
                        )
                    )
                    print(spk2)
                with open(args.sample_file, "a") as f:
                    f.write(f"\n\n\n{i_epoch} epoch, {i_batch} batch:\n")
                    for dialogue, spk2 in zip(input_ids_decoded, spk2_reply_decoded):
                        write_buffer = "\n#########################\n"
                        write_buffer += (
                            dialogue.replace(tokenizer.eos_token, "\n").replace(
                                tokenizer.pad_token, ""
                            )
                        ) + "\n"
                        write_buffer += spk2 + "\n"
                        f.write(write_buffer)
            if i_batch % args.step_save == 0:
                torch.save(
                    chatbot, os.path.join(args.model_folder, f"{i_epoch}epoch.bin")
                )

<<<<<<< HEAD
=======

>>>>>>> 4c9672d0847152fb808b125e45cfde77cd3a863a
def analyze_engaging(chatbot, interlocutor, tokenizer, data_loader, args, args_bot):
    chatbot.eval()
    interlocutor.eval()

    i_batch = 0
    num_batch = 30
    repeat_time = 10
<<<<<<< HEAD
    engaging_scores = [[0 for _ in range(repeat_time)] for _ in range(args.batch_size*num_batch)]
=======
    engaging_scores = [
        [0 for _ in range(repeat_time)] for _ in range(args.batch_size * num_batch)
    ]
>>>>>>> 4c9672d0847152fb808b125e45cfde77cd3a863a
    with torch.no_grad():
        for batch in tqdm(data_loader):
            for i_time in range(repeat_time):
                input_ids, mask = batch.copy()
<<<<<<< HEAD
                chatbot_reply, log_prob, coherence_score, coherence_loss = generate_response(
                    input_ids, mask, tokenizer, chatbot, args_bot,
                    model_2=interlocutor, device=args_bot.device
=======
                (
                    chatbot_reply,
                    log_prob,
                    coherence_score,
                    coherence_loss,
                ) = generate_response(
                    input_ids,
                    mask,
                    tokenizer,
                    chatbot,
                    args_bot,
                    model_2=interlocutor,
                    device=args_bot.device,
>>>>>>> 4c9672d0847152fb808b125e45cfde77cd3a863a
                )

                # padding reply
                max_l = max((len(reply) for reply in chatbot_reply))
                reply_tensor = []
                for reply in chatbot_reply:
                    reply_tensor.append(
                        [tokenizer.eos_token_id]
                        + reply
                        + [tokenizer.pad_token_id for _ in range(max_l - len(reply))]
                    )
                reply_tensor = torch.tensor(reply_tensor)
                input_ids = torch.cat((input_ids, reply_tensor), 1)
<<<<<<< HEAD
                mask_append = torch.ones((args.batch_size, max_l + 1))  # +1 for eos_token
                mask = torch.cat((mask, mask_append), 1)

                spk2_reply, _, coherence_score_spk2, coherence_loss_spk2 = generate_response(
                    input_ids, mask, tokenizer, interlocutor, args_bot,
                    device=args_bot.device_2
=======
                mask_append = torch.ones(
                    (args.batch_size, max_l + 1)
                )  # +1 for eos_token
                mask = torch.cat((mask, mask_append), 1)

                (
                    spk2_reply,
                    _,
                    coherence_score_spk2,
                    coherence_loss_spk2,
                ) = generate_response(
                    input_ids,
                    mask,
                    tokenizer,
                    interlocutor,
                    args_bot,
                    device=args_bot.device_2,
>>>>>>> 4c9672d0847152fb808b125e45cfde77cd3a863a
                )

                # get engagin score
                q_r = [[q, r] for q, r in zip(chatbot_reply, spk2_reply)]
                score = get_score(q_r, tokenizer)
                for j, sc in enumerate(score):
                    engaging_scores[i_batch * args.batch_size + j][i_time] = sc
<<<<<<< HEAD
                
                input_ids_decoded = tokenizer.batch_decode(input_ids)
                spk2_reply_decoded = tokenizer.batch_decode(spk2_reply)

                with open("analysis/dialogpt_engaging_30bt_10t_sentences.txt", "a") as f:
                    f.write(f"\{i_batch} batch:\n")
                    for dialogue, spk2, sc in zip(input_ids_decoded, spk2_reply_decoded, score):
=======

                input_ids_decoded = tokenizer.batch_decode(input_ids)
                spk2_reply_decoded = tokenizer.batch_decode(spk2_reply)

                with open(
                    "analysis/dialogpt_engaging_30bt_10t_sentences.txt", "a"
                ) as f:
                    f.write(f"\{i_batch} batch:\n")
                    for dialogue, spk2, sc in zip(
                        input_ids_decoded, spk2_reply_decoded, score
                    ):
>>>>>>> 4c9672d0847152fb808b125e45cfde77cd3a863a
                        write_buffer = f"\n#########################\n score: {sc}\n"
                        write_buffer += (
                            dialogue.replace(tokenizer.eos_token, "\n").replace(
                                tokenizer.pad_token, ""
                            )
                        ) + "\n"
                        write_buffer += spk2 + "\n"
                        f.write(write_buffer)
            i_batch += 1
            if i_batch >= num_batch:
                break
    repeat_avg_scores = []
    repeat_std_scores = []
    for scores in engaging_scores:
        avg_score = sum(scores) / len(scores)
        repeat_avg_scores.append(avg_score)
        std_tmp = 0
        for sc in scores:
            std_tmp += (sc - avg_score) ** 2
<<<<<<< HEAD
        repeat_std_scores.append((std_tmp / repeat_time)**0.5)
    with open("analysis/dialogpt_engaging_30bt_10t_scores.txt", "w") as f:
        f.write(f"#### DialoGPT\n30 batch, repeat 10 times\n")
        f.write(f"\ntotal mean scores: {sum(repeat_avg_scores) / len(repeat_avg_scores)}\n")
=======
        repeat_std_scores.append((std_tmp / repeat_time) ** 0.5)
    with open("analysis/dialogpt_engaging_30bt_10t_scores.txt", "w") as f:
        f.write(f"#### DialoGPT\n30 batch, repeat 10 times\n")
        f.write(
            f"\ntotal mean scores: {sum(repeat_avg_scores) / len(repeat_avg_scores)}\n"
        )
>>>>>>> 4c9672d0847152fb808b125e45cfde77cd3a863a
        f.write("\naverage in each repeat\n")
        for avg_sc in repeat_avg_scores:
            f.write(f"{avg_sc}  ")
        f.write("\n")
        f.write("\nstd in each repeat\n")
        for std_sc in repeat_std_scores:
            f.write(f"{std_sc}  ")
        f.write("\n")

    print(engaging_scores)
    print(repeat_avg_scores)
    print(repeat_std_scores)
<<<<<<< HEAD
                
=======
>>>>>>> 4c9672d0847152fb808b125e45cfde77cd3a863a


class ARG_BOT:
    def __init__(self, bt=8):
        self.dataset_path = "data/personachat_self_original.json"
        self.dataset_cache = "data/dataset_cache"

        # how many sentences in history, start from interlocutor
        # ..., interlocutor, chatbot
        self.history_turn = 3

        self.history_max_length = 80
        self.device = "cuda:0"
        self.device_2 = "cuda:0"
        self.no_sample = False
        self.max_length = 30
        self.min_length = 1
        self.seed = 2
        self.temperature = 0.9
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
<<<<<<< HEAD
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save_dir", type=str, default="model/")
    parser.add_argument("--model_name", type=str, default="dialogpt_1turn_lr1-5_w02")
=======
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--save_dir", type=str, default="model/")
    parser.add_argument("--model_name", type=str, default="dialogpt_1turn_lr1-6_w02")
>>>>>>> 4c9672d0847152fb808b125e45cfde77cd3a863a

    # parameters
    parser.add_argument("--weight_coherence", type=float, default=0.2)

    # steps
    parser.add_argument("--step_optimize", type=int, default=2)
    parser.add_argument("--step_sample", type=int, default=200)
    parser.add_argument("--step_save", type=int, default=1000)

    args = parser.parse_args()
    args_bot = ARG_BOT(args.batch_size)

    args.model_folder = os.path.join(args.work_space, args.save_dir, args.model_name)
    args.sample_file = f"sample/sample_{args.model_name}.txt"
    os.makedirs(args.model_folder, exist_ok=True)

    # ===== prepare dataset, models and optimizer ==========
    # "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    chatbot = AutoModelForCausalLM.from_pretrained(args.model_checkpoint)
    interlocutor = AutoModelForCausalLM.from_pretrained(args.model_checkpoint)

    chatbot.to(args_bot.device)
    interlocutor.to(args_bot.device_2)
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token = tokenizer.decode([0])

    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders_DialoGPT(
        args_bot, tokenizer
    )
    del train_sampler, valid_sampler

<<<<<<< HEAD
    # analyze_engaging(chatbot, interlocutor, tokenizer, val_loader, args, args_bot)

    train(chatbot, interlocutor, tokenizer, train_loader, args, args_bot)
=======
    analyze_engaging(chatbot, interlocutor, tokenizer, val_loader, args, args_bot)

    # train(chatbot, interlocutor, tokenizer, train_loader, args, args_bot)
>>>>>>> 4c9672d0847152fb808b125e45cfde77cd3a863a


if __name__ == "__main__":
    main()
