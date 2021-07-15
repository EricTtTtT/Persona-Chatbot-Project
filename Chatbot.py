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
    get_data_loaders,
)

from Engaging_classifier import analyze_engagement
from Persona_Selector import prepare_persona_selector, select_persona
from tensorboardX import SummaryWriter

# from collections import defaultdict
# from torch.utils.data import DataLoader, TensorDataset
# from utils import get_dataset, download_pretrained_model, top_filtering

writer = SummaryWriter("runs")
SPECIAL_TOKENS = ["<bos>", "<|eos|>", "<speaker1>", "<speaker2>", "<pad>"]


def generate_response(personality, history, tokenizer, model, arg, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    bos, eos, speaker1, speaker2, pad = special_tokens_ids
    if current_output is None:
        current_output = [[] for _ in range(arg.train_batch_size)]

    sequence_bt = [
        [[bos] + persona_i] + history_i
        for persona_i, history_i in zip(personality, history)
    ]
    sequence_bt = [
        [seq[0]]
        + [
            [speaker2 if (len(seq) - i) % 2 else speaker1] + s
            for i, s in enumerate(seq[1:])
        ]
        for seq in sequence_bt
    ]
    token_type_ids_bt = [
        [speaker2 if i % 2 else speaker1 for i, s in enumerate(seq) for _ in s]
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
    ).to(arg.device)
    token_type_ids_bt = pad_sequence(
        [torch.LongTensor(x) for x in token_type_ids_bt],
        batch_first=True,
        padding_value=speaker1,
    ).to(arg.device)
    mask = pad_sequence(
        [torch.LongTensor(x) for x in mass], batch_first=True, padding_value=0
    ).to(arg.device)

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
        probs = torch.softmax(logits, dim=-1)

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


def prepare_chatbot(check_point, bt=8, root="."):
    class ARG:
        def __init__(self):
            self.dataset_path = os.path.join(
                root, "data/personachat_self_original.json"
            )
            self.dataset_cache = os.path.join(root, "data/dataset_cache")
            self.max_history = 2
            self.num_candidates = 1
            self.device = "cuda"
            self.no_sample = False
            self.max_length = 40
            self.min_length = 1
            self.seed = 2
            self.temperature = 1
            self.top_k = 0
            self.top_p = 0
            self.distributed = False
            self.personality_permutations = 1
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
    parser.add_argument(
        "--model_name", type=str, default="new_b16_l-5_d2"
    )  # persona selector model folder
    parser.add_argument("--load_model_path", type=str, default="")
    parser.add_argument("--log_step", type=int, default=2)
    parser.add_argument("--print_sample_step", type=int, default=20)
    parser.add_argument("--save_time_step", type=int, default=100)
    parser.add_argument("--select", type=bool, default=True)
    parser.add_argument("--fix", type=bool, default=False)

    args = parser.parse_args()
    os.makedirs(
        os.path.join(args.work_space, args.save_dir, args.model_name), exist_ok=True
    )
    log_file_path = os.path.join(args.work_space, f"record/{args.model_name}.txt")

    # ===== prepare dataset, models and optimizer ==========
    model, interlocutor, tokenizer, arg = prepare_chatbot(
        os.path.join(args.work_space, args.model_checkpoint), bt=args.batch_size
    )
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(
        arg, tokenizer
    )
    del val_loader, train_sampler, valid_sampler
    print("\n\nlen(train_loader): ", len(train_loader), "\n\n")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=6732
    )
    bert_model.train()

    persona_selector, persona_pool = prepare_persona_selector(
        load_path=args.load_model_path
    )
    # optimizer = torch.optim.Adam(persona_selector.id_selector.parameters(), lr = args.lr)
    optimizer = AdamW(bert_model.parameters(), lr=1e-5, eps=1e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=500, num_training_steps=8000
    )  # PyTorch scheduler

    selector_history = 5

    print(
        """
        ######################################################
        finish preparing  !!!!!!!!!!!!!!!!!
        ######################################################"""
    )
    persona_selector.train()
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
            # ===== select persona for interlocutor ========
            # ===== generate s0 from dataset ===============
            interlocutor_persona_enc = []
            history_enc = []
            score_record = []
            for input_i in input_ids:
                persona = []
                history = []
                sen = []
                per = True
                sen_spk2 = True
                #  print(input_i)
                for i in range(1, len(input_i[0])):
                    if input_i[0][i] != 50258:  # <|eos|>
                        if per:
                            if input_i[0][i] == 50261:  # <speaker2>
                                per = False
                            else:
                                persona.append(int(input_i[0][i]))
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
                interlocutor_persona_enc.append(persona)
                history_enc.append(history)
            # print(history)
            # print('######################')
            # print('persona')
            # print(tokenizer.decode(interlocutor_persona_enc[0]))
            # print('######################')
            # print('history init')
            # for sen_enc in history_enc[0]:
            #     print(tokenizer.decode(sen_enc))

            # ===== generate s1 from interlocutor ==========
            with torch.no_grad():
                response_enc = generate_response(
                    interlocutor_persona_enc, history_enc, tokenizer, interlocutor, arg
                )
                for i in range(args.batch_size):
                    history_enc[i].append(response_enc[i])

            value_init = get_score([h[-2:] for h in history_enc], tokenizer)

            # ===== select persona for s2 ==================
            history = [
                [tokenizer.decode(s) for s in h[-selector_history:]]
                for h in history_enc
            ]

            if args.select:
                persona_s2, log_prob = select_persona(
                    persona_selector, persona_pool, history, bert_tokenizer, bert_model
                )
            else:
                persona_s2, log_prob = random.sample(persona, args.batch_size), [
                    0 for _ in range(args.batch_size)
                ]
            prob_record.append(log_prob)

            # ===== generate s2 ============================
            persona_enc = [
                tokenizer.encode_plus(p_i, return_attention_mask=False)["input_ids"]
                for p_i in persona_s2
            ]
            with torch.no_grad():
                response_enc = generate_response(
                    persona_enc, history_enc, tokenizer, model, arg
                )
                for i in range(args.batch_size):
                    history_enc[i].append(response_enc[i])

            # ===== generate s3 from interlocutor ==========
            with torch.no_grad():
                response_enc = generate_response(
                    interlocutor_persona_enc, history_enc, tokenizer, interlocutor, arg
                )
                for i in range(args.batch_size):
                    history_enc[i].append(response_enc[i])

            score_record.append(get_score([h[-2:] for h in history_enc], tokenizer))

            # ===== select persona for s4 ==================
            history = [
                [tokenizer.decode(s) for s in h[-selector_history:]]
                for h in history_enc
            ]
            if not args.fix:
                if args.select:
                    persona_s4, log_prob = select_persona(
                        persona_selector,
                        persona_pool,
                        history,
                        bert_tokenizer,
                        bert_model,
                    )
                else:
                    persona_s4, log_prob = random.sample(
                        persona_pool, args.batch_size
                    ), [0 for _ in range(args.batch_size)]
            else:
                persona_s4, log_prob = persona_s2, [0 for _ in range(args.batch_size)]
            prob_record.append(log_prob)

            # ===== generate s4 ============================
            persona_enc = [
                tokenizer.encode_plus(p_i, return_attention_mask=False)["input_ids"]
                for p_i in persona_s4
            ]
            with torch.no_grad():
                response_enc = generate_response(
                    persona_enc, history_enc, tokenizer, model, arg
                )
                for i in range(args.batch_size):
                    history_enc[i].append(response_enc[i])

            # ===== generate s5 from interlocutor ==========
            with torch.no_grad():
                response_enc = generate_response(
                    interlocutor_persona_enc, history_enc, tokenizer, interlocutor, arg
                )
                for i in range(args.batch_size):
                    history_enc[i].append(response_enc[i])

            score_record.append(get_score([h[-2:] for h in history_enc], tokenizer))

            # print('value_init\n', value_init)
            # print('prob_record\n', prob_record)
            # print('score_record\n', score_record)

            rewards_s4 = [score_record[1][i] for i in range(args.batch_size)]
            rewards_s2 = [score_record[0][i] for i in range(args.batch_size)]

            rewards_ori = torch.tensor([rewards_s2, rewards_s4], device=arg.device)

            # print(rewards_ori)
            rewards = []

            for r in rewards_ori:
                rewards.append((r - r.mean()))
            # rewards = (rewards_ori - rewards_ori.mean()) / (rewards_ori.std() + 1e-9)

            loss = 0
            # rewards = rewards[:args.batch_size//2]
            # prob_record = prob_record[:args.batch_size//2]
            # print('rewards\n', rewards)
            # rewards = rewards[:1]
            # prob_record = prob_record[:1]
            for r, l in zip(rewards, prob_record):
                loss -= sum(r * l)
                break

            rewards_ori = rewards_ori.detach().cpu().numpy()
            rw_2, rw_4 = np.sum(rewards_ori, axis=1)
            sc_2, sc_4 = np.sum(score_record, axis=1)
            sc_ori = np.sum(value_init)

            # optimizer.zero_grad()
            loss /= 2
            # persona_selector.train()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
            # optimizer.step()
            # scheduler.step()

            i_batch += 1
            sc += rewards_ori[0].mean()
            print("loss", loss.item(), "reward:", rewards_ori.mean())
            if i_batch % 2 == 0:
                torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if i_batch % args.log_step == 0:
                niter = i_epoch * len(train_loader) + i_batch
                writer.add_scalar("Train/Loss", loss, niter)
                writer.add_scalar("Train/Score_origin", sc_ori, niter)
                writer.add_scalar("Train/Score_s2", sc_2, niter)
                writer.add_scalar("Train/Score_s4", sc_4, niter)
                writer.add_scalar("Train/Reward_s2", rw_2, niter)
                writer.add_scalar("Train/Reward_s4", rw_4, niter)

            if i_batch % args.print_sample_step == 0:
                with open(log_file_path, "a") as fp:
                    fp.write("\n===== dialogue sample ======================\n")
                    fp.write(
                        f"persona_interlocutor :\n{tokenizer.decode(interlocutor_persona_enc[0])}\n"
                    )
                    fp.write(f"persona_s2: {persona_s2[0]}\n")
                    fp.write(f"persona_s4: {persona_s4[0]}\n")
                    fp.write(f"\nhistory + s1~s5 at {i_epoch} epoch {i_batch} batch\n")
                    for sen_enc in history_enc[0]:
                        fp.write(f"{tokenizer.decode(sen_enc)}\n")
                print("===== print dialogue sample ==================")
                print(
                    f"\npersona_interlocutor :\n{tokenizer.decode(interlocutor_persona_enc[0])}\n"
                )
                print("\npersona_s2\n", persona_s2[0])
                print("\npersona_s4\n", persona_s4[0])
                print(f"\nhistory + s1~s5 at {i_epoch} epoch {i_batch} batch")
                for sen_enc in history_enc[0]:
                    print(tokenizer.decode(sen_enc))
                print(sc / args.print_sample_step)
                sc = 0

            if i_batch % args.save_time_step == 0:
                torch.save(
                    persona_selector,
                    os.path.join(
                        args.work_space,
                        args.save_dir,
                        args.model_name,
                        f"{i_epoch}_epoch.pkl",
                    ),
                )

            prob_record.clear()
            score_record.clear()
        torch.save(
            persona_selector,
            os.path.join(
                args.work_space, args.save_dir, args.model_name, f"{i_epoch}_epoch.pkl"
            ),
        )


if __name__ == "__main__":
    main()
