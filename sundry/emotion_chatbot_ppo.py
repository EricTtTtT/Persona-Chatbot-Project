# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
import logging
import re
import warnings
import matplotlib.pyplot as plt
from os.path import join
from itertools import chain
from argparse import ArgumentParser

from pprint import pformat, pprint
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertModel, BertTokenizer, BertForSequenceClassification, AutoModel


from torch.nn.utils.rnn import pad_sequence

from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_, get_data_loaders_persona

from Engaging_classifier import analyze_engagement
from Persona_Selector import prepare_persona_selector, select_persona
from tensorboardX import SummaryWriter
import wandb
from PPO_emo_ac import PPO
from remove_duplicate_persona import remove_duplicate_persona

# from collections import defaultdict
# from torch.utils.data import DataLoader, TensorDataset
# from utils import get_dataset, download_pretrained_model, top_filtering

from GoEmotions_pytorch.model import EmoBertForMultiLabelClassification
from GoEmotions_pytorch.multilabel_pipeline import MultiLabelPipeline


writer = SummaryWriter("runs")
SPECIAL_TOKENS = ["<bos>", "<|eos|>", "<speaker1>", "<speaker2>", "<pad>"]


def generate_response(personality, history, tokenizer, model, arg):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    bos, eos, speaker1, speaker2, pad = special_tokens_ids

    sequence = [[[bos] + persona_i] + history_i for persona_i, history_i in zip(personality, history)]
    sequence = [
        [seq[0]] + [[speaker2 if (len(seq) - i) % 2 else speaker1] + s for i, s in enumerate(seq[1:])] for seq in sequence
    ]
    token_type_ids = [[speaker2 if i % 2 else speaker1 for i, s in enumerate(seq) for _ in s] for seq in sequence]
    sequence = [list(chain(*seq)) for seq in sequence]
    mask_len = [len(x) for x in sequence]
    mass = []
    for i in range(len(sequence)):
        m = [1 for j in range(mask_len[i])]
        mass.append(m[:])

    sequence = pad_sequence(
        [torch.LongTensor(x) for x in sequence], batch_first=True, padding_value=tokenizer.encode("<pad>")[0]
    ).to(arg.device)
    token_type_ids = pad_sequence([torch.LongTensor(x) for x in token_type_ids], batch_first=True, padding_value=speaker1).to(
        arg.device
    )
    mask = pad_sequence([torch.LongTensor(x) for x in mass], batch_first=True, padding_value=0).to(arg.device)

    _, past = model(sequence, attention_mask=mask, token_type_ids=token_type_ids)

    token_tp = torch.LongTensor([[speaker2] if len(x) % 2 else [speaker1] for x in history]).to(arg.device)
    prev = torch.LongTensor([[speaker2] if len(x) % 2 else [speaker1] for x in history]).to(arg.device)

    temp_sen = [[] for i in range(len(history))]
    for i_word in range(arg.max_length):
        logits, past = model(prev, token_type_ids=token_tp, past=past)
        logits = logits.squeeze(0).squeeze(1)
        # logits = top_filtering(logits, top_k=arg.top_k, top_p=arg.top_p)
        probs = torch.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1]
        # prev = [torch.topk(prob_i, 1)[1] if arg.no_sample else torch.multinomial(prob_i, 1) for prob_i in probs]
        for prev_i, prob_i in zip(prev, probs):
            if i_word < arg.min_length and prev_i.item() in special_tokens_ids:
                while prev_i.item() in special_tokens_ids:
                    if prob_i.max().item() == 1:
                        warnings.warn("Warning: model generating special token with probability 1.")
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


def prepare_chatbot(check_point, bt=8, root=".", seed=1000):
    class ARG:
        def __init__(self):
            self.dataset_path = os.path.join(root, "data/personachat_self_original.json")
            self.dataset_cache = os.path.join(root, "data/cache_persona_3his")
            self.history_turn = 3
            self.history_max_length = 50
            self.persona_max_length = 30
            self.device = "cuda:0"
            self.no_sample = False
            self.max_length = 40
            self.min_length = 1
            self.seed = seed
            self.temperature = 1
            self.top_k = 0
            self.top_p = 0
            self.distributed = False
            self.personality_permutations = 1
            self.local_rank = -1
            self.train_batch_size = bt
            self.valid_batch_size = bt
            # print("batch_size is ", self.train_batch_size)
            # exit(0)

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
    # model = model_class.from_pretrained(check_point)
    model = model_class.from_pretrained(check_point)
    # model = torch.load(check_point)
    interlocutor = model_class.from_pretrained(check_point)
    model.to(arg.device).eval()
    interlocutor.to(arg.device).eval()

    add_special_tokens_(model, tokenizer)
    add_special_tokens_(interlocutor, tokenizer)

    return model, interlocutor, tokenizer, arg


# def get_score(history_enc_last_two, tokenizer):
#     # query = []
#     # reply = []
#     # for history_enc_i in history_enc_last_two:
#     #     query.append(tokenizer.decode(history_enc_i[0]))
#     #     reply.append(tokenizer.decode(history_enc_i[1]))
#     # score = analyze_engagement(query, reply)
#     score = [len(h[1]) for h in history_enc_last_two]
#     return score


def main():
    parser = ArgumentParser()
    parser.add_argument("--work_space", type=str, default=".")
    parser.add_argument("--model_checkpoint", type=str, default="model/gpt2_persona_model/")
    parser.add_argument("--save_dir", type=str, default="Emo_saved_model/")
    parser.add_argument("--model_name", type=str, default="exc_joy__turn1__lr5_u2_bt32")

    # hyper parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--ac_lr", type=float, default=3e-5)
    parser.add_argument("--cr_lr", type=float, default=0.005)
    parser.add_argument("--turn", type=int, default=1)
    parser.add_argument("--K_epoch", type=int, default=3)
    parser.add_argument("--sample_size", type=int, default=10)

    # steps
    parser.add_argument("--step_sample", type=int, default=200)
    parser.add_argument("--step_save", type=int, default=1000)
    parser.add_argument("--step_update", type=int, default=10)

    args = parser.parse_args()
    # args.model_save_folder = os.path.join(args.work_space, args.save_dir, args.model_name)
    # os.makedirs(args.model_save_folder, exist_ok=True)
    args.sample_file = os.path.join(args.work_space, f"sample/{args.model_name}.txt")

    # ===== prepare dataset, models and optimizer ==========
    model, interlocutor, tokenizer, arg = prepare_chatbot(
        os.path.join(args.work_space, args.model_checkpoint), bt=args.batch_size, seed=9999
    )
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders_persona(arg, tokenizer)
    del train_sampler, valid_sampler

    # persona_pool = remove_duplicate_persona()
    persona_pool = np.load("./clean_persona.npy")
    print("shape of persona_pool", np.shape(persona_pool))

    # bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(persona_pool))
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model.train()

    # ====== emotion score ==========
    emo_tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-group")
    emo_model = EmoBertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-group")

    goemotions = MultiLabelPipeline(model=emo_model, tokenizer=emo_tokenizer, threshold=0.3)

    ppo = PPO(
        persona_pool=persona_pool,
        bert_model=bert_model,
        lr_actor=float(args.ac_lr),
        lr_critic=float(args.cr_lr),
        sample_size=args.sample_size,
        K_epochs=args.K_epoch,
        seed=arg.seed,
    )
    # wandb.init(project="persona_chatbot", entity="erictien")
    print("Seed is ", arg.seed)
    wandb.init(
        project="give_me_entropy",
        entity="persona_chatbot_ntuee",
        config={
            "batch_size": args.batch_size,
            "epoch": args.epoch,
            "lr_actor": ppo.lr_actor,
            "lr_critic": ppo.lr_critic,
            "turn": args.turn,
            "sample_iter": args.sample_size,
            "step_update": args.step_update,
            "K_epochs": ppo.K_epochs,
            "random_seed": arg.seed,
        },
    )

    # wandb.config = {
    #     "batch_size": args.batch_size,
    #     "epoch": args.epoch,
    #     "lr_actor": ppo.lr_actor,
    #     "lr_critic": ppo.lr_critic,
    #     "turn": args.turn,
    #     "sample_iter": args.sample_size,
    #     "step_update": args.step_update,
    #     "K_epochs": ppo.K_epochs

    # }
    # persona_selector.id_selector.train()
    # persona_selector.train()
    # optimizer = torch.optim.Adam(persona_selector.parameters(), lr = args.lr)
    # # optimizer = AdamW(bert_model.parameters(), lr=1e-5, eps=1e-5)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=500, num_training_steps=8000
    # )  # PyTorch scheduler

    print(
        """
        ######################################################
        finish preparing  !!!!!!!!!!!!!!!!!
        ######################################################"""
    )

    i_batch = 0
    for i_epoch in range(args.epoch):
        for inter_persona_ori, history_ori, len_p, len_h in tqdm(train_loader):
            # recover inter_persona and history from padded datum
            inter_persona_enc = []
            for p_ori, lens in zip(inter_persona_ori, len_p):
                l = sum(lens)
                inter_persona_enc.append(p_ori[:l].tolist())
            history_enc = []
            for h_ori, lens in zip(history_ori, len_h):
                tmp = []
                j = 0
                for l in lens:
                    if l > 0:
                        tmp.append((h_ori[j : j + l]).tolist())
                        j += l
                history_enc.append(tmp)
            save_history_enc = history_enc
            for k in range(args.K_epoch):
                for t in range(args.sample_size):
                    # print("T is ", t)
                    persona_bot_record = []
                    history_enc = save_history_enc
                    for i_turn in range(args.turn):
                        # get chatbot persona
                        history = [[tokenizer.decode(s) for s in h] for h in history_enc]
                        persona_bot = ppo.select_action(history, bert_tokenizer)
                        # print("Buffer size ", ppo.buffer.size())
                        persona_bot_enc = [tokenizer.encode(p) for p in persona_bot]
                        persona_bot_record.append(persona_bot)

                        # generate one turn response
                        with torch.no_grad():
                            # chatbot
                            response_enc = generate_response(persona_bot_enc, history_enc, tokenizer, model, arg)
                            history_enc = [h + [r] for h, r in zip(history_enc, response_enc)]

                            # interlocutor
                            response_enc = generate_response(inter_persona_enc, history_enc, tokenizer, interlocutor, arg)
                            history_enc = [h + [r] for h, r in zip(history_enc, response_enc)]
                        try_text = [tokenizer.decode(h[-1]) for h in history_enc]
                        ppo.buffer.rewards.append(goemotions.get_positive_score(try_text))

                    # if i_batch % args.step_update == 0:
                    loss, critic_loss, entropy, reward = ppo.update_writer(turn=args.turn, accum_size=args.sample_size)
                    wandb.log({"loss": loss, "loss_critic": critic_loss, "reward": reward, "entropy": entropy}, step=i_batch)
                    # if i_batch % (args.step_update * args.sample_size) == 0:
                    i_batch += 1
                ppo.update()
                if i_batch % args.step_sample == 0:
                    for j in range(args.batch_size):
                        sample_str = "\n#########################\n"
                        sample_str += "interlocutor persona:  " + tokenizer.decode(inter_persona_enc[j]) + "\n"
                        for k in range(args.turn):
                            sample_str += f"chatbot persona {k}:  {persona_bot_record[k][j]} \n"
                        for h in history_enc[j]:
                            sample_str += tokenizer.decode(h) + "\n"
                        sample_str += "\n"
                    print(sample_str)
                    with open(args.sample_file, "a") as f:
                        f.write(f"\n\n\n{i_epoch} epoch, {i_batch} batch:\n")
                        f.write(sample_str)

                if i_batch % args.step_save == 0:
                    torch.save(ppo.policy, os.path.join(ppo.output_dir, "model.bin"))
            print("***************************************")
            print("Start Validation")
            print("***************************************")
            with torch.no_grad():
                valid_reward = []
                for inter_persona_ori, history_ori, len_p, len_h in tqdm(val_loader):
                    # recover inter_persona and history from padded datum
                    inter_persona_enc = []
                    for p_ori, lens in zip(inter_persona_ori, len_p):
                        l = sum(lens)
                        inter_persona_enc.append(p_ori[:l].tolist())
                    history_enc = []
                    for h_ori, lens in zip(history_ori, len_h):
                        tmp = []
                        j = 0
                        for l in lens:
                            if l > 0:
                                tmp.append((h_ori[j : j + l]).tolist())
                                j += l
                        history_enc.append(tmp)
                    for i_turn in range(args.turn):
                        # get chatbot persona
                        history = [[tokenizer.decode(s) for s in h] for h in history_enc]
                        persona_bot = ppo.select_action(history, bert_tokenizer, valid=True)
                        # print("Buffer size ", ppo.buffer.size())
                        persona_bot_enc = [tokenizer.encode(p) for p in persona_bot]
                        persona_bot_record.append(persona_bot)

                        # generate one turn response
                        # with torch.no_grad():
                        # chatbot
                        response_enc = generate_response(persona_bot_enc, history_enc, tokenizer, model, arg)
                        history_enc = [h + [r] for h, r in zip(history_enc, response_enc)]

                        # interlocutor
                        response_enc = generate_response(inter_persona_enc, history_enc, tokenizer, interlocutor, arg)
                        history_enc = [h + [r] for h, r in zip(history_enc, response_enc)]
                        try_text = [tokenizer.decode(h[-1]) for h in history_enc]
                        # print("Try text is ", try_text)
                        valid_reward.append(np.mean(goemotions.get_positive_score(try_text)))
                    # exit(0)
            wandb.log({"valid_reward": np.mean(valid_reward), "valid_reward_std": np.std(valid_reward)}, step=i_batch)


if __name__ == "__main__":
    main()
