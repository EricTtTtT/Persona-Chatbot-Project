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
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup, AutoModelForCausalLM, AutoTokenizer

from train import get_data_loaders_DialoGPT, get_data_loaders

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

        prev = torch.LongTensor([[tokenizer.eos_token_id] for _ in range(bt)]).to(device)
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

            prev = torch.multinomial(logits, num_samples=1)
            # log_p = [math.log(p[idx]) for idx, p in zip(prev, logits)]

            # if model_2:
            #     # get average of probs_2
            #     probs_2 = []
            #     for j in range(bt):
            #         if cont[j]:
            #             probs_2.append(logits_2[j][prev[j].item()].item())
            #     if len(probs_2) == 0:
            #         avg_prob_2 = 0
            #     else:
            #         avg_prob_2 = sum(probs_2) / len(probs_2)

            if i_word == 0:
                for j in range(bt):
                    # print("\n##########")
                    # print(prev[j].item(), logits[j][prev[j].item()].item(), logits_2[j][prev[j].item()].item())
                    temp_sen[j].append(prev[j].item())
                    # log_prob[j].append(log_p[j])
                    # if model_2:
                    #     tmp_loss = F.cross_entropy(logits[j].unsqueeze(0), prev.view(-1)[j].unsqueeze(0))
                    #     coherence_score_arr[j].append(logits_2[j][prev[j].item()].item())
                    #     coherence_loss[j] += (logits_2[j][prev[j].item()].item() - avg_prob_2) * tmp_loss
                continue

            for j in range(bt):
                if cont[j]:
                    if temp_sen[j][-1] != eos_id:
                        temp_sen[j].append(prev[j].item())
                        # log_prob[j].append(log_p[j])
                        # if model_2:
                        #     tmp_loss = F.cross_entropy(logits[j].unsqueeze(0), prev.view(-1)[j].unsqueeze(0))
                        #     coherence_score_arr[j].append(logits_2[j][prev[j].item()].item())
                        #     coherence_loss[j] += (logits_2[j][prev[j].item()].item() - avg_prob_2) * tmp_loss
                    else:
                        cont[j] = False
            if not (True in cont):
                break

        for i in range(bt):
            if temp_sen[i][-1] == eos_id:
                temp_sen[i][:] = temp_sen[i][:-1]
                # log_prob[i][:] = log_prob[i][:-1]

        # coherence_score = [0 for _ in range(bt)]
        # if model_2:
        #     for j, scores in enumerate(coherence_score_arr):
        #         if len(scores) != 0:
        #             coherence_score[j] = sum(scores) / len(scores)

        return temp_sen
        # return temp_sen, log_prob, coherence_score, coherence_loss


def eval(chatbot, interlocutor, tokenizer, train_loader, args, args_bot):
    chatbot.eval()
    interlocutor.eval()

    # optimizer = AdamW(chatbot.parameters(), lr=1e-5, eps=1e-5)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=2000, num_training_steps=8000
    # )
    record = {}
    i = 0
    for batch in tqdm(train_loader):
        if i < 3:
            i += 1
        else:
            break
        input_ids, _ = batch
        print(np.shape(input_ids))
        print(tokenizer.batch_decode(input_ids))
        print(np.shape(tokenizer.batch_decode(input_ids)))
        sentences = tokenizer.batch_decode(input_ids)[0].split("<|endoftext|> ")
        print(np.shape(sentences))
        for sen in sentences:
            if sen not in record.keys():
                record[sen] = []
    print(record.keys())
    exit(0)
    dict_keys(
        [
            "hello what are doing today?",
            "i just got done watching a horror movie",
            "wow! i do love a good horror movie. loving this cooler weather",
            "yes! my son is in junior high and i just started letting him watch them too",
            "neat!! i used to work in the human services field",
            "yes i bet you can get hurt. my wife works and i stay at home",
            "hi! how are you doing tonight?",
            "great. in my spare time i do volunteer work.",
            "i work in a homeless shelter in my town.",
            "cool. not really into cars. my day job is wrestling.",
            "yes, i love the crowds, getting to know people.",
            "hello, how are you doing?",
            "that is great, me too! i'm married and my husband and i've 2 children.",
            "no, we recently purchased a new house, so we cannot afford it. have you?",
            "i enjoy going to concerts, i see the rolling stones every year.",
            "maybe you should consider going back to school. i did. i major in economics.",
            "hi how are you doing? i am okay how about you?",
            "i used to do home health aide but now i am disabled.",
            "i graduated and got my license worked a while and became i'll.",
            "i have dogs and i walk them. and a cat.",
            "i enjoy reading read about michael jackson and psychology today.",
            "hi i dye my hair 4 times a year",
            "no i do not, i've a dream, it is to work from home",
            "that is really cool that you can do that",
            "no i don't, i faint when i see blood",
            "i went to school to be a vet, but i didn't like it",
            "hi! do you like turtles?",
            "i have a turtle his name is speedy. kitties are nice too, tho!",
            "what are your kitties names?",
            "i like that! i go to preschool.",
            "how old are you? i turned four on my birthday!",
            "hello, how are you tonight? do you have pink and blue hair?",
            "what do you like to do in your spare time? i bird watch.",
            "those are fun. i've a cat, do you?",
        ]
    )
    # optimizer.zero_grad()
    record = {}
    data = [
        "hello what are doing today?",
        "i just got done watching a horror movie",
        "wow! i do love a good horror movie. loving this cooler weather",
        "yes! my son is in junior high and i just started letting him watch them too",
        "neat!! i used to work in the human services field",
        "yes i bet you can get hurt. my wife works and i stay at home",
        "hi! how are you doing tonight?",
        "great. in my spare time i do volunteer work.",
        "i work in a homeless shelter in my town.",
        "cool. not really into cars. my day job is wrestling.",
        "yes, i love the crowds, getting to know people.",
        "hello, how are you doing?",
        "that is great, me too! i'm married and my husband and i've 2 children.",
        "no, we recently purchased a new house, so we cannot afford it. have you?",
        "i enjoy going to concerts, i see the rolling stones every year.",
        "maybe you should consider going back to school. i did. i major in economics.",
        "hi how are you doing? i am okay how about you?",
        "i used to do home health aide but now i am disabled.",
        "i graduated and got my license worked a while and became i'll.",
        "i have dogs and i walk them. and a cat.",
        "i enjoy reading read about michael jackson and psychology today.",
        "hi i dye my hair 4 times a year",
        "no i do not, i've a dream, it is to work from home",
        "that is really cool that you can do that",
        "no i don't, i faint when i see blood",
        "i went to school to be a vet, but i didn't like it",
        "hi! do you like turtles?",
        "i have a turtle his name is speedy. kitties are nice too, tho!",
        "what are your kitties names?",
        "i like that! i go to preschool.",
        "how old are you? i turned four on my birthday!",
        "hello, how are you tonight? do you have pink and blue hair?",
        "what do you like to do in your spare time? i bird watch.",
        "those are fun. i've a cat, do you?",
    ]
    for i in range(0, len(data), 8):
        input_ids = tokenizer.batch_encode(data[i : i + 8])
        print(np.shape(input_ids))
        print("batch is ", input_ids)
        for sen in input_ids:
            print("sen is ", sen)
            if sen not in record.keys():
                record[sen] = []
        # exit(0)
        chatbot_reply = generate_response(
            input_ids, mask, tokenizer, chatbot, args_bot, model_2=interlocutor, device=args_bot.device
        )

        # padding reply
        max_l = max((len(reply) for reply in chatbot_reply))
        reply_tensor = []
        for reply in chatbot_reply:
            reply_tensor.append([tokenizer.eos_token_id] + reply + [tokenizer.pad_token_id for _ in range(max_l - len(reply))])
        reply_tensor = torch.tensor(reply_tensor)
        input_ids = torch.cat((input_ids, reply_tensor), 1)
        mask_append = torch.ones((args.batch_size, max_l + 1))  # +1 for eos_token
        mask = torch.cat((mask, mask_append), 1)

        spk2_reply = generate_response(input_ids, mask, tokenizer, interlocutor, args_bot, device=args_bot.device_2)

        # get engagin score
        q_r = [[q, r] for q, r in zip(chatbot_reply, spk2_reply)]
        scores = get_score(q_r, tokenizer)
        print("score is ", scores)
        print(record.keys())
        for dialog, score in zip(tokenizer.batch_decode(input_ids), scores):
            print(dialog.split("<|endoftext|>")[0])
            record[dialog.split("<|endoftext|>")[0] + "<|endoftext|>"].append(score)
        import json

        with open("gpt_record.json", "w") as f:
            json.dumps(record, f)
        # score_mean = sum(score) / len(score)
        # running_score += score_mean

        # running_coherence += sum(coherence_score) / len(coherence_score)

        # # calculate reward
        # rewards = [sc - score_mean for sc in score]
        # # rewards = [sc - score_mean + coh_sc*args.weight_coherence for sc, coh_sc in zip(score, coherence_score)]
        # for r, log_p, coh_loss in zip(rewards, log_prob, coherence_loss):
        #     if len(log_p) == 0:
        #         logging.info("Error! divided by 0 due to empty log_p")
        #         continue
        #     running_loss -= r * sum(log_p) / len(log_p)
        #     running_loss += coh_loss * args.weight_coherence

        # i_batch += 1
        # if i_batch % args.step_optimize == 0:
        #     # loss = torch.tensor(
        #     #     running_loss / args.step_optimize, requires_grad=True
        #     # )
        #     loss = running_loss.clone().detach().requires_grad_(True) / args.step_optimize
        #     loss.backward()

        #     torch.nn.utils.clip_grad_norm_(chatbot.parameters(), .5)
        #     optimizer.step()
        #     scheduler.step()
        #     optimizer.zero_grad()

        #     writer.add_scalar("Train/Loss", loss, i_iter)
        #     writer.add_scalar(
        #         "Train/Engaging_score", running_score / args.step_optimize, i_iter
        #     )
        #     writer.add_scalar(
        #         "Train/Coherence_score", running_coherence / args.step_optimize, i_iter
        #     )
        #     i_iter += 1

        #     running_loss = 0
        #     running_score = 0
        #     running_coherence = 0

        # if i_batch % args.step_sample == 0:
        input_ids_decoded = tokenizer.batch_decode(input_ids)
        spk2_reply_decoded = tokenizer.batch_decode(spk2_reply)
        print(np.shape(input_ids_decoded))
        for dialogue, spk2 in zip(input_ids_decoded, spk2_reply_decoded):
            print("\n#########################")
            print("dialogue is : ", dialogue.replace(tokenizer.eos_token, "\n").replace(tokenizer.pad_token, ""))
            print("spk2 is : ", spk2)
        exit(0)
        # with open(args.sample_file, "a") as f:
        #     f.write(f"\n\n\n{i_epoch} epoch, {i_batch} batch:\n")
        #     for dialogue, spk2 in zip(input_ids_decoded, spk2_reply_decoded):
        #         write_buffer = "\n#########################\n"
        #         write_buffer += (
        #             dialogue.replace(tokenizer.eos_token, "\n").replace(
        #                 tokenizer.pad_token, ""
        #             )
        #         ) + "\n"
        #         write_buffer += spk2 + "\n"
        #         f.write(write_buffer)
        # if i_batch % args.step_save == 0:
        #     torch.save(
        #         chatbot, os.path.join(args.model_folder, f"{i_epoch}epoch.bin")
        #     )


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
    parser.add_argument("--model_checkpoint", type=str, default="model/dialogpt-medium/")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--save_dir", type=str, default="model/")
    parser.add_argument("--model_name", type=str, default="dialogpt_1turn_lr1-6_w02")

    # parameters
    parser.add_argument("--weight_coherence", type=float, default=0.2)

    # steps
    parser.add_argument("--step_optimize", type=int, default=2)
    parser.add_argument("--step_sample", type=int, default=200)
    parser.add_argument("--step_save", type=int, default=1000)

    args = parser.parse_args()
    args_bot = ARG_BOT(args.batch_size)

    args.model_folder = os.path.join(args.work_space, args.save_dir, args.model_name)
    args.sample_file = f"sample/eval_{args.model_name}.txt"
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

    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args_bot, tokenizer)
    # train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders_DialoGPT(
    #     args_bot, tokenizer
    # )
    del train_sampler, valid_sampler

    eval(chatbot, interlocutor, tokenizer, val_loader, args, args_bot)


if __name__ == "__main__":
    main()
