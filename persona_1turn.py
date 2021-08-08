# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
import logging
import warnings
import matplotlib.pyplot as plt
from os.path import join
from itertools import chain
from argparse import ArgumentParser

from pprint import pformat
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from transformers import (
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    BertModel,
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from torch.nn.utils.rnn import pad_sequence

from train import (
    SPECIAL_TOKENS,
    build_input_from_segments,
    add_special_tokens_,
    get_data_loaders_persona,
)

from Engaging_classifier import analyze_engagement
from Persona_Selector import prepare_persona_selector, select_persona
from tensorboardX import SummaryWriter

# from collections import defaultdict
# from torch.utils.data import DataLoader, TensorDataset
# from utils import get_dataset, download_pretrained_model, top_filtering

writer = SummaryWriter("runs")
SPECIAL_TOKENS = ["<bos>", "<|eos|>", "<speaker1>", "<speaker2>", "<pad>"]


def generate_response(persona, history, length, tokenizer, model, arg):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    bos, eos, speaker1, speaker2, pad = special_tokens_ids

    print(persona)
    print(history)
    print(length)
    exit()

    # build input sequence

    """
    _, past = model(sequence_bt, attention_mask=mask, token_type_ids=token_type_ids_bt)

    token_tp = torch.LongTensor(
        [[speaker2] if len(x) % 2 else [speaker1] for x in history]
    ).to(arg.device)
    prev = torch.LongTensor(
        [[speaker2] if len(x) % 2 else [speaker1] for x in history]
    ).to(arg.device)

    temp_sen = [[] for i in range(len(history))]
    for i_word in range(arg.max_length):
        logits, past = model(prev, token_type_ids=token_tp, past=past)
        logits = logits.squeeze(0).squeeze(1)
        # logits = top_filtering(logits, top_k=arg.top_k, top_p=arg.top_p)

        # Apply temperature
        # probs = torch.softmax(logits, dim=-1)
        probs = torch.softmax(logits / arg.temperature, dim=-1)

        prev = torch.multinomial(probs, num_samples=1)
        # prev = [torch.topk(prob_i, 1)[1] if arg.no_sample else torch.multinomial(prob_i, 1) for prob_i in probs]
        for prev_i, prob_i in zip(prev, probs):
            if i_word < arg.min_length and prev_i.item() in special_tokens_ids:
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
    """


def train(chatbot, interlocutor, tokenizer, train_loader, args, args_bot):
    chatbot.train()
    interlocutor.eval()

    optimizer = AdamW(chatbot.parameters(), lr=args.lr, eps=1e-5)
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
            persona, history, length = batch
            (
                chatbot_reply,
                log_prob,
                coherence_score,
                coherence_loss,
            ) = generate_response(
                persona,
                history,
                length,
                tokenizer,
                chatbot,
                args_bot,
                model_2=interlocutor,
                device=args_bot.device,
            )
            exit()

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
                loss = (
                    running_loss.clone().detach().requires_grad_(True)
                    / args.step_optimize
                )
                loss.backward()

                torch.nn.utils.clip_grad_norm_(chatbot.parameters(), 0.5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                writer.add_scalar("Train/Loss", loss, i_iter)
                writer.add_scalar(
                    "Train/Engaging_score", running_score / args.step_optimize, i_iter
                )
                writer.add_scalar(
                    "Train/Coherence_score",
                    running_coherence / args.step_optimize,
                    i_iter,
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


def prepare_chatbot(check_point, bt=8, root="."):
    class ARG:
        def __init__(self):
            self.dataset_path = os.path.join(
                root, "data/personachat_self_original.json"
            )
            self.dataset_cache = os.path.join(root, "data/dataset_cache")
            self.history_turn = 3
            self.history_max_length = 80
            self.device = "cuda:0"
            self.device_2 = "cuda:0"
            self.no_sample = False
            self.max_length = 40
            self.min_length = 1
            self.seed = 2
            self.temperature = 0.5
            self.top_k = 0
            self.top_p = 0
            self.distributed = False
            self.local_rank = -1
            self.train_batch_size = bt
            self.valid_batch_size = bt

    arg = ARG()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)

    random.seed(arg.seed)
    torch.random.manual_seed(arg.seed)
    torch.cuda.manual_seed(arg.seed)

    logger.info("Get pretrained model and tokenizer")
    tokenizer_class, model_class = (
        GPT2Tokenizer,
        GPT2LMHeadModel,
    )  # if args.model == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
    tokenizer = tokenizer_class.from_pretrained(check_point)
    model = model_class.from_pretrained(check_point)
    interlocutor = model_class.from_pretrained(check_point)
    model.to(arg.device).eval()
    interlocutor.to(arg.device).eval()

    add_special_tokens_(model, tokenizer)
    add_special_tokens_(interlocutor, tokenizer)

    return model, interlocutor, tokenizer, arg


def get_score(history_enc_last_two, tokenizer):
    query = []
    reply = []
    for history_enc_i in history_enc_last_two:
        query.append(tokenizer.decode(history_enc_i[0]))
        reply.append(tokenizer.decode(history_enc_i[1]))
    score = analyze_engagement(query, reply)
    return score


def main():
    parser = ArgumentParser()
    parser.add_argument("--work_space", type=str, default=".")
    parser.add_argument(
        "--model_checkpoint", type=str, default="model/gpt2_persona_model/"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--save_dir", type=str, default="model/")
    parser.add_argument("--model_name", type=str, default="dialogpt_1turn_lr1-6_w02")

    # parameters
    parser.add_argument("--weight_coherence", type=float, default=0.2)

    # steps
    parser.add_argument("--step_optimize", type=int, default=2)
    parser.add_argument("--step_sample", type=int, default=200)
    parser.add_argument("--step_save", type=int, default=1000)

    args = parser.parse_args()
    args.model_folder = os.path.join(args.work_space, args.save_dir, args.model_name)
    args.sample_file = f"sample/sample_{args.model_name}.txt"
    os.makedirs(args.model_folder, exist_ok=True)

    # ===== prepare dataset, models and optimizer ==========
    chatbot, interlocutor, tokenizer, args_bot = prepare_chatbot(
        os.path.join(args.work_space, args.model_checkpoint), bt=args.batch_size
    )
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders_persona(
        args_bot, tokenizer
    )
    del val_loader, train_sampler, valid_sampler

    # bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # bert_model = BertModel.from_pretrained("bert-base-uncased")
    # bert_model = BertForSequenceClassification.from_pretrained(
    #     "bert-base-uncased", num_labels=6732
    # )
    # bert_model.train()

    # persona_selector, persona_pool = prepare_persona_selector(
    #     load_path=args.load_model_path
    # )

    train(chatbot, interlocutor, tokenizer, train_loader, args, args_bot)


if __name__ == "__main__":
    main()
