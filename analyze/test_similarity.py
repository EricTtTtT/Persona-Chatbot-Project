import os
import random
import matplotlib.pyplot as plt
from os.path import join
from argparse import ArgumentParser

from pprint import pformat
from numpy.core.fromnumeric import mean
from tqdm import tqdm
import numpy as np
import torch

from transformers import BertModel, BertTokenizer, BertForSequenceClassification

from train import get_data_loaders

from Chatbot import *


def analyze():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="model/gpt2_persona_model/")
    parser.add_argument("--save_dir", type=str, default="model/")
    parser.add_argument("--model_name", type=str, default="new_b16_l-5_d2")  # persona selector model folder
    parser.add_argument("--load_model_path", type=str, default="")
    parser.add_argument("--work_space", type=str, default=".")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--fix", type=bool, default=False)

    args = parser.parse_args()
    log_file_path = os.path.join(args.work_space, f"record/analysis.txt")

    # ===== prepare dataset, models and optimizer ==========
    model, interlocutor, tokenizer, arg = prepare_chatbot(
        os.path.join(args.work_space, args.model_checkpoint), bt=args.batch_size
    )
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(arg, tokenizer)
    del train_loader, train_sampler, valid_sampler
    print("\n\nlen(val_loader): ", len(val_loader), "\n\n")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6732)
    bert_model.eval()
    persona_selector, persona_pool = prepare_persona_selector(load_path=args.load_model_path)

    print(
        """
        ######################################################
        finish preparing  !!!!!!!!!!!!!!!!!
        ######################################################"""
    )

    with torch.no_grad():
        count = 0
        total_rw = {}
        for input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids in tqdm(val_loader):
            init = True
            count += 1
            for i in range(10):
                for i_p, persona_i in enumerate(persona_pool[:30]):
                    # print(f"{i_p}: {persona_i}")
                    persona_bot = [persona_i for _ in range(args.batch_size)]
                    persona_bot_enc = [
                        tokenizer.encode_plus(p_i, return_attention_mask=False)["input_ids"] for p_i in persona_bot
                    ]

                    score_record = []
                    # ===== select persona for interlocutor ========
                    # ===== generate s0 from dataset ===============
                    interlocutor_persona_enc = []
                    history_enc = []
                    score_record = []
                    for input_i in input_ids:
                        # print(input_i)
                        # print(tokenizer.decode(input_i))
                        # total_rw[tokenizer.decode(input_i)] = {}
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
                        # print(tokenizer.decode(history[0]))

                    # print(history)
                    # print('######################')
                    # print('interlocutor\'s persona')
                    # print(tokenizer.decode(interlocutor_persona_enc[0]))
                    # print('######################')
                    # print('history init')
                    # for sen_enc in history_enc[0]:
                    #     print(tokenizer.decode(sen_enc))

                    # ===== generate s1 from interlocutor ==========
                    if init:
                        for i in range(args.batch_size):
                            if tokenizer.decode(history_enc[i][0]) not in total_rw.keys():
                                total_rw[tokenizer.decode(history_enc[i][0])] = {}
                        init = False
                    response_enc = generate_response(interlocutor_persona_enc, history_enc, tokenizer, interlocutor, arg)
                    for i in range(args.batch_size):
                        history_enc[i].append(response_enc[i])

                    # value_init = get_score([h[-2:] for h in history_enc], tokenizer)
                    # print(f"\t S2 score is {value_init}")
                    # ===== generate s2 ============================
                    response_enc = generate_response(persona_bot_enc, history_enc, tokenizer, model, arg)
                    for i in range(args.batch_size):
                        history_enc[i].append(response_enc[i])
                    for i in range(args.batch_size):
                        # print(tokenizer.decode(history_enc[i][0]))
                        if persona_i not in total_rw[tokenizer.decode(history_enc[i][0])].keys():
                            total_rw[tokenizer.decode(history_enc[i][0])][persona_i] = []
                        total_rw[tokenizer.decode(history_enc[i][0])][persona_i].append(tokenizer.decode(response_enc[i]))

                    # ===== generate s3 from interlocutor ==========
                    response_enc = generate_response(interlocutor_persona_enc, history_enc, tokenizer, interlocutor, arg)
                    for i in range(args.batch_size):
                        history_enc[i].append(response_enc[i])

                    score = get_score([h[-2:] for h in history_enc], tokenizer)
                    score_record.append(score)
                    # print(len(score))
                    # exit(0)
                    # print(np.shape([h[-2:] for h in history_enc]))
                    # print(exit(0))

                    # for i in args.batch_size:
                    # print([h[-2:] for h in history_enc])
                    # print(f"\t S2 score is {score}")
                    # ===== generate s4 ============================
                    response_enc = generate_response(persona_bot_enc, history_enc, tokenizer, model, arg)
                    for i in range(args.batch_size):
                        history_enc[i].append(response_enc[i])

                    for i in range(args.batch_size):
                        total_rw[tokenizer.decode(history_enc[i][0])][persona_i].append(tokenizer.decode(response_enc[i]))
                    # ===== generate s5 from interlocutor ==========
                    response_enc = generate_response(interlocutor_persona_enc, history_enc, tokenizer, interlocutor, arg)
                    for i in range(args.batch_size):
                        history_enc[i].append(response_enc[i])
                    score = get_score([h[-2:] for h in history_enc], tokenizer)
                    # print(f"\t S4 score is {score}")
                    score_record.append(score)
                    # print(tokenizer.decode(history_enc[i][0]))
                    # ==== calculating rewards =====================
                    # rewards_s4 = [score_record[1][i] for i in range(args.batch_size)]
                    # rewards_s2 = [score_record[0][i] for i in range(args.batch_size)]
                    # rw2 = mean(rewards_s2)
                    # rw4 = mean(rewards_s4)
                    # print('mean(rewards_s2)', rw2)
                    # print('mean(rewards_s4)', rw4)

                    # total_rw[persona_i] = {}
                    # for i in range(args.batch_size):
                    #     total_rw[persona_i][i] = [score_record[0][i], score_record[1][i]]

            # exit(0)
            if count == 10:
                # from sentence_transformers import SentenceTransformer
                # model = SentenceTransformer('bert-base-nli-mean-tokens')
                record = {}
                for key in total_rw.keys():
                    if key not in record.keys():
                        record[key] = {0: {}, 1: {}}
                    # print("******************************************")
                    #     print("input : \t ", key)
                    for persona in total_rw[key].keys():
                        for i in range(len(total_rw[key][persona])):
                            if persona not in record[key][int(i % 2)].keys():
                                record[key][int(i % 2)][persona] = []
                            record[key][int(i % 2)][persona].append(total_rw[key][persona][i])

                #         print(f"\t {persona} : \t {total_rw[key][persona]}")
                #     print("******************************************")
                # print(total_rw)
                import json

                with open(f"ori_sentence_05.json", "w") as file:
                    json.dump(total_rw, file)
                with open("sentence_05.json", "w") as file:
                    json.dump(record, file)
                exit(0)
            # with open("all_persona_rewards.csv", "a") as f:
            #     f.write(f"{total_rw[0][0]:.3f}, {total_rw[0][1]:.3f}")
            #     for rw in total_rw[1:]:
            #         f.write(f", {rw[0]:.3f}, {rw[0]:.3f}")
            #     f.write("\n")


if __name__ == "__main__":
    analyze()
