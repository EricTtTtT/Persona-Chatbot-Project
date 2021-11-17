# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
import logging
import re
from typing import Mapping
import warnings
import matplotlib.pyplot as plt
from os.path import join
from itertools import chain
from argparse import ArgumentParser

from pprint import pformat
from numpy.core.fromnumeric import mean
from numpy.ma import core
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from transformers import GPT2LMHeadModel,GPT2Tokenizer,BertModel,BertTokenizer,BertForSequenceClassification, AutoModel, AdamW
import json

from torch.nn.utils.rnn import pad_sequence

from train import (
    SPECIAL_TOKENS,
    build_input_from_segments,
    add_special_tokens_,
    get_data_loaders,
)

# from Engaging_classifier import analyze_engagement
from Persona_Selector import prepare_persona_selector, select_persona
from tensorboardX import SummaryWriter
from PPO import PPO
from remove_duplicate_persona import remove_duplicate_persona
# from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.dataset import Dataset
from remove_duplicate_persona import remove_duplicate_persona
# from utils import get_dataset, download_pretrained_model, top_filtering

writer = SummaryWriter("runs")
SPECIAL_TOKENS = ["<bos>", "<|eos|>", "<speaker1>", "<speaker2>", "<pad>"]

class dataset(Dataset):

    def __init__(self, data, label, weight):
        self.data = data
        self.label = label
        self.weight = weight
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.weight[idx]
class Model(torch.nn.Module):

    def __init__(self, bert_model):
        super(Model, self).__init__()
        self.net = bert_model
        self.last_layer = torch.nn.Sigmoid()
        # self.last_layer = torch.nn.Softmax()
        # self.net = torch.nn.Sequential(
        #     self.bert_model,
        #     self.last_layer
        # )

    def forward(self, **batch):
        output = self.net(**batch, output_hidden_states=True)
        return self.last_layer(output[0])
        # self.optimizer = AdamW()
    
def reshape_data(args, map_table):

    dim = 6732
    persona_data_path = "./data/personachat_self_original.json"
    persona_data_file = open(persona_data_path)
    persona_data = json.load(persona_data_file)

    # ==========read persona sentences=====================
    data_type_list = ["train", "valid"]
    persona_set = set()
    for data_type in data_type_list:
        count = 0
        for i in range(len(persona_data[data_type])):
            count += len(persona_data[data_type][i]["personality"])
            for i_sentence in persona_data[data_type][i]["personality"]:
                persona_set.add(i_sentence)
        print(data_type, "data size: ", count)
    print("total # of persona: ", len(persona_set))
    persona_pool = sorted(list(persona_set))
    persona_table = {}
    for i in  range(len(persona_pool)):
        persona_table[persona_pool[i]] = i
    
    def get_labels(set):
        labels = [[0.0 for j in range(dim)] for i in range(len(set))]
        for i in range(len(set)):
            for p in set[i]['personality']:
                # map table will map the delete persona to it corresponding persona
                labels[i][persona_table[map_table.get(p)]] = 1.0
        return torch.tensor(labels)      
    def process_set(set):
        new_set = []
        for i in range(len(set)):
            count = 0
            sen = "[CLS] "
            for s in set[i]['utterances'][3]['history']:
                if count % 2 == 1:
                    sen += s + " [SEP] "
                count += 1
            new_set.append(sen)
        return new_set
    train_set = process_set(persona_data['train'])
    valid_set = process_set(persona_data['valid'])
    np.save(args.data_dir + "new_train_set.npy", train_set)
    np.save(args.data_dir + "new_vallid_set.npy", valid_set)
    # exit(0)
    train_labels = get_labels(persona_data['train'])
    val_labels = get_labels(persona_data['valid'])

    return train_set, valid_set, train_labels, val_labels
    
def prepare_data_loader(args, train_valid_bound = 0.75):
    # ==========setting IO=================================
    
    train_set= np.load(args.data_dir + "new_train_set.npy").tolist()
    valid_set= np.load(args.data_dir + "new_valid_set.npy").tolist()
    train_labels = torch.load(args.data_dir + "train_label.pt")
    valid_labels = torch.load(args.data_dir + "valid_label.pt")

    # for data in map_table:
        
    train_weights = None
    valid_weights = None
    if os.path.isdir(args.data_dir + f"alpha_{args.alpha}"):
        train_weights = torch.load(args.data_dir + f"alpha_{args.alpha}/" + "train_weight.pt")
        valid_weights = torch.load(args.data_dir + f"alpha_{args.alpha}/" + "valid_weight.pt")
    else:
        print("******************************")
        print("Prepare for weights ....")
        print("******************************")
        print("")
        os.makedirs(args.data_dir + f"alpha_{args.alpha}")
        train_weights = torch.tensor(get_weight(train_labels, args.alpha).tolist())
        valid_weights = torch.tensor(get_weight(valid_labels, args.alpha).tolist())
        torch.save(train_weights, args.data_dir + f"alpha_{args.alpha}/" + "train_weight.pt")
        torch.save(valid_weights, args.data_dir + f"alpha_{args.alpha}/" + "valid_weight.pt")
        print("")
        print("******************************")
        print("Finish prepare weights")
        print("******************************")
        print("")

    # It seems like part of persona only shown in validation set
    # So we will split train & val set into new train & val set
    train_bound = int(len(train_set) * train_valid_bound)
    valid_bound = int(len(valid_set) * train_valid_bound)

    # Split train set into new train set
    new_train_set = train_set[:train_bound]
    new_train_labels = train_labels[:train_bound]
    new_train_weights = train_weights[:train_bound]
    
    # Split vlaid set into new train set
    new_train_set.extend(valid_set[:valid_bound])
    new_train_labels = torch.cat((new_train_labels, valid_labels[:valid_bound]), dim = 0)
    new_train_weights = torch.cat((new_train_weights ,valid_weights[:valid_bound]), dim = 0)
    
    # Split train set into new val set
    new_valid_set = train_set[train_bound:]
    new_valid_labels = train_labels[train_bound:]
    new_valid_weights = train_weights[train_bound:]

    # Split vlaid set into new val set
    new_valid_set.extend(valid_set[valid_bound:])
    new_valid_labels = torch.cat((new_valid_labels, valid_labels[valid_bound:]), dim = 0)
    new_valid_weights = torch.cat((new_valid_weights ,valid_weights[valid_bound:]), dim = 0)

    train_data = dataset(new_train_set, new_train_labels, new_train_weights)
    valid_data = dataset(new_valid_set, new_valid_labels, new_valid_weights)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # exit(0)
    # return new_train_set, new_valid_set, new_train_labels, new_valid_labels, new_train_weights, new_valid_weights
    return train_loader, valid_loader
    # Below are old version of load data
    # Save for future use

    dim = 6732
    persona_data_path = "./data/personachat_self_original.json"
    persona_data_file = open(persona_data_path)
    persona_data = json.load(persona_data_file)

    # ==========read persona sentences=====================
    data_type_list = ["train", "valid"]
    persona_set = set()
    for data_type in data_type_list:
        count = 0
        for i in range(len(persona_data[data_type])):
            count += len(persona_data[data_type][i]["personality"])
            for i_sentence in persona_data[data_type][i]["personality"]:
                persona_set.add(i_sentence)
        print(data_type, "data size: ", count)
    print("total # of persona: ", len(persona_set))
    persona_pool = sorted(list(persona_set))
    persona_table = {}
    for i in  range(len(persona_pool)):
        persona_table[persona_pool[i]] = i
    
    # def get_labels(set):
    #     labels = [[0.0 for j in range(dim)] for i in range(len(set))]
    #     for i in range(len(set)):
    #         for p in set[i]['personality']:
    #             labels[i][persona_table[p]] = 1.0
    #     return torch.tensor(labels)      
    # def process_set(set):
    #     new_set = []
    #     for i in range(len(set)):
    #         sen = "[CLS] "
    #         for s in set[i]['utterances'][1]['history']:
    #             sen += s + " [SEP] "
    #         new_set.append(sen)
    #     return new_set
    # train_set = process_set(persona_data['train'])
    # valid_set = process_set(persona_data['valid'])
    
    # train_labels = get_labels(persona_data['train'])
    # val_labels = get_labels(persona_data['valid'])
    

# function to train the model
def get_persona(idxs, idx2persona_pool):
    print(" ")
    for idx in idxs:
        print("\t", idx2persona_pool[idx])
def calculate_acc(pred, ans, idx2persona_pool = None, threshold = 0.5):
    
    correct_0 = 0
    correct_1 = 0
    num_0 = 0
    num_1 = 0
    for i in  range(len(pred)):
        # if idx2persona_pool is not None:
        #     print("Ans persona are : ")
        #     get_persona(ans[i].nonzero(as_tuple=True)[0], idx2persona_pool)
        #     print("Predict persona are : ")
        for j in range(len(pred[i])):
            
            if ans[i][j] == 0:
                num_0 += 1
                if pred[i][j] < 0.5:
                    correct_0 += 1
            else:
                num_1 += 1
                if pred[i][j] > 0.5:
                    correct_1 += 1            
    # print(f"In {(len(pred) * len(pred[0]))} test, we success to get {correct} right")
    return correct_0 / num_0, correct_1 / num_1

# loss_func = torch.nn.CrossEntropyLoss()
def get_weight(labels, alpha = 0.0005):
    weights = np.full((len(labels), len(labels[0])), alpha)
    for i in range(len(labels)):
        for j in range(len(labels[0])):
            if labels[i][j] == 1.0:
                weights[i][j] = (1-alpha)
    return torch.tensor(weights)
import matplotlib.pyplot as plt
def draw(args, data, epoch, type=""):
    os.makedirs(args.record_dir + f"{type}", exist_ok=True)
    if type == "hit" or "recalls":
        
        hit, recall, mmr = data
        K = epoch
        for k in K:
            plt.plot(hit[k], label = f"hit@{k}")
        plt.ylabel("%")
        plt.legend(loc="best")
        plt.title("Hit@K")
        plt.savefig(args.record_dir + f"{type}/hit.jpg")
        plt.clf()
        
        for k in K:
            plt.plot(recall[k], label = f"recall@{k}")
        plt.ylabel("%")
        plt.legend(loc="best")
        plt.title("Recall@K")
        plt.savefig(args.record_dir + f"{type}/recall.jpg")
        plt.clf()
        for k in K:
            plt.plot(mmr[k], label = f"MMR@{k}")
        plt.ylabel("%")
        plt.legend(loc="best")
        plt.title("MMR@K")
        plt.savefig(args.record_dir + f"{type}/MMR.jpg")
        plt.clf()
        with open(args.record_dir + f"{type}/record.txt", "w") as f:
            for k in K:
                f.writelines(f"hit@{k} is {np.mean(hit[k])}% " )
                f.writelines("\n")
                f.writelines(f"recall@{k} is {np.mean(recall[k])}% " )
                f.writelines("\n")
                f.writelines(f"MMR@{k} is {np.mean(mmr[k])}% " )
                f.writelines("\n")
    else:
        loss_record, acc0_record, acc1_record = data
        plt.plot(loss_record, label="loss")
        plt.legend(loc="best")
        plt.title("loss")
        plt.savefig(args.record_dir + f"{type}/epoch_{epoch}_loss.jpg")
        plt.clf()
        plt.plot(acc0_record, label="acc_0")
        plt.plot(acc1_record, label="acc_1")
        plt.legend(loc="best")
        plt.ylabel("%")
        plt.title("acc")
        plt.savefig(args.record_dir + f"{type}/epoch_{epoch}_acc.jpg")
        plt.clf()
def train(model, tokenizer, train_loader, optimizer, args, device, epoch):
    loss_record = []
    acc0_record = []
    acc1_record = []
    model.net.train()
    

    # iterate over batches
    for idx, batchs in enumerate(train_loader):

        batch, labels, weights = batchs
        # push the batch to gpu
        # print(type(batch))
        # print(type(batch[0]))
        labels = labels.to(device)
        # new_labels = [[]for i in range(len(labels))]
        # labels = labels.non_zero(as_tuple = True)
        # for j, i in enumerate(labels[0]):
        #     new_labels[i].append(labels[1][j])
        # encode a batch
        loss_func = torch.nn.BCELoss(weight=weights.to(device))
        encode_input = tokenizer(batch,
        add_special_tokens=False,
        truncation=True,
        padding=True,
        return_tensors="pt",
        ).to(device)
        # clear previously calculated gradients 
        model.net.zero_grad()        

        # get model predictions for the current batch
        # output = self.actor(**state, output_hidden_states=True)
        preds = model(**encode_input)
        # preds = torch.multinomial(preds, min(torch.count_nonzero(labels, dim =1)))
        # compute the loss between actual and predicted values
        # for j in  range(batch_size):
        loss = loss_func(preds, labels) + args.reg_cof * (torch.mean(preds) - torch.mean(labels) - torch.std(preds))
        # loss = loss_func(preds, labels) 
        # exit(0)
        # add on to the total loss
        optimizer.zero_grad()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds=preds.detach().cpu().numpy()
        loss = loss.detach().cpu()
        encode_input = encode_input.to(torch.device("cpu"))
        labels = labels.cpu()
        weights = weights.cpu()
        loss_record.append(loss)

        # progress update after every 5 batches.
        if idx % 5 == 0 :
            print('  Batch {:>5,}  of  {:>5,}.'.format(idx, len(train_loader)))
            print("loss is ", loss)
            if epoch > 0:
                pool = list(np.load(args.data_dir + "clean_persona_pool.npy"))
                acc_0, acc_1 = calculate_acc(preds, labels, pool)
            else:
                acc_0, acc_1 = calculate_acc(preds, labels)
            print(f"Accuracy of 0 is {acc_0*100}%")
            print(f"Accuracy of 1 is {acc_1*100}%")
            acc0_record.append(100*acc_0)
            acc1_record.append(100*acc_1)
        # append the model predictions
    # Record the loss and acc
    if args.write:
        print("Will write to files")
        draw(args, (loss_record, acc0_record, acc1_record), epoch, type="train")
    return model
    

def evaluate(model, tokenizer, valid_loader, args, device, epoch, idx2persona_pool):
    loss_record = []
    acc0_record = []
    acc1_record = []
    model.net.eval()

    # iterate over batches
    for batch, labels, weights in valid_loader:

        # push the batch to gpu
        print("In the evaluation ....")
        # batch = valid_set[i:i+args.batch_size]
        labels = labels.to(device)
        # encode a batch
        loss_func = torch.nn.BCELoss(weight=weights.to(device))
        encode_input = tokenizer(batch,
        add_special_tokens=False,
        truncation=True,
        padding=True,
        return_tensors="pt",
        ).to(device)
        # clear previously calculated gradients 

        # get model predictions for the current batch
        # output = self.actor(**state, output_hidden_states=True)
        preds = model(**encode_input)
        # compute the loss between actual and predicted values
        # for j in  range(batch_size):
        loss = loss_func(preds, labels.type(torch.float)) + args.reg_cof * torch.mean(preds)
        loss = loss.detach().cpu()
        labels = labels.cpu()
        # exit(0)
        # add on to the total loss
        # progress update after every 5 batches.
        # if i % (batch_size * 5) == 0 :
        #     print('  Batch {:>5,}  of  {:>5,}.'.format(i, len(valid_set)))
        print("loss is ", loss)
        acc_0, acc_1 = calculate_acc(preds, labels, list(idx2persona_pool))
        print(f"Accuracy of 0 is {acc_0*100}%")
        print(f"Accuracy of 1 is {acc_1*100}%")
        acc0_record.append(100*acc_0)
        acc1_record.append(100*acc_1)
        preds=preds.detach().cpu().numpy()
        weights = weights.cpu()
        encode_input = encode_input.to(torch.device("cpu"))
        loss_record.append(loss)



    # Record the loss and acc
    if args.write :
        print("Will write to file")
        draw(args, (loss_record, acc0_record, acc1_record), epoch, type="valid")
    return np.mean(loss_record), np.mean(acc1_record)

    # function for evaluating the model
    
def calculate_hit(preds, labels, K):
    
    
    total = 0
    hit = {}
    recall = {}
    mmr = {}
    for k in K:
        hit[k] = 0
        recall[k] = 0
        mmr[k] = 0
    for label, pred in zip(labels, preds):
        new_label = (label == 1).nonzero(as_tuple=True)[0]
        total += new_label.size()[0]
        for l in new_label:
            for k in K:
                if l in pred[:k]:
                    hit[k] += 1
                    recall[k] += 1
                    mmr[k] = 1/((pred[:k] == l).nonzero(as_tuple=True)[0][0]+1)
     
    for k in K:
        hit[k] *= 100
        hit[k] /= total
        
        recall[k] *= 100
        recall[k] /= (k*labels.size()[0])

        mmr[k] *= 100
    return hit, recall, mmr
            
def hit(model, tokenizer, valid_loader, device, args, K = [10, 100, 200]):
    
    model.eval()
    hits = {}
    recalls = {}
    MMr = {}
    for k in K:
        hits[k] = []
        recalls[k] = []
        MMr[k] = []
    for batch, labels, weights in valid_loader:

        # push the batch to gpu
        print("In the Hit@K ....")
        # batch = valid_set[i:i+args.batch_size]
        # labels = labels.to(device)
        # encode a batch
        loss_func = torch.nn.BCELoss(weight=weights.to(device))
        encode_input = tokenizer(batch,
        add_special_tokens=False,
        truncation=True,
        padding=True,
        return_tensors="pt",
        ).to(device)
        
        # Index is what we want
        _, indexs = torch.sort(model(**encode_input).cpu(), descending=True)
        
        hit, recall, mmr = calculate_hit(indexs, labels, K)
        for k in K:
            hits[k].append(hit[k]) 
            recalls[k].append(recall[k]) 
            MMr[k].append(mmr[k])
        print("")
        print("***********************")
        for k in K:
            print(f"hit@{k} is {hits[k][-1]}%")
            print(f"recall@{k} is {recalls[k][-1]}%")
            print(f"MMR@{k} is {MMr[k][-1]}%")
        
        print("***********************")
        
        weights = weights.cpu()
        encode_input = encode_input.to(torch.device("cpu"))
    # Record the loss and acc
    if args.write :
        print("Will write to file")
        draw(args, (hits, recalls, MMr), K, type="hit")
        
    
def main():
    parser = ArgumentParser()
    parser.add_argument("--work_space", type=str, default=".")
    parser.add_argument(
        "--model_checkpoint", type=str, default="model/gpt2_persona_model/"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.003)
    parser.add_argument("--reg_cof", type=float, default=0.001)
    parser.add_argument("--write",  action='store_true', default=False)
    parser.add_argument("--save",  action='store_true', default=False)
    parser.add_argument("--eval",  action='store_true', default=False)
    parser.add_argument("--hit",  type=int, default = None, nargs='+')
    parser.add_argument("--data_dir", type=str, default="./Pretrain_Dir/data/")
    parser.add_argument("--model_dir", type=str, default="./Pretrain_Dir/checkpoint/")
    parser.add_argument("--record_dir", type=str, default="./Pretrain_Dir/record/")
    parser.add_argument("--load_path", type=str, default="./Pretrain_Dir/checkpoint/new_std_clean_lr_0.0001_epoch_10_batchsize_16_alpha_0.0023_regcof_0.001_/")
    
    args = parser.parse_args()

    dir = f"new_std_clean_lr_{args.lr}_epoch_{args.epoch}_batchsize_{args.batch_size}_alpha_{args.alpha}_regcof_{args.reg_cof}_"
    args.record_dir += dir
    args.model_dir += dir
    if args.eval or args.hit is not None:
        args.record_dir = args.load_path.replace("checkpoint", "record")
    # args.record_dir += f"new_std_clean_lr_{args.lr}_epoch_{args.epoch}_batchsize_{args.batch_size}_alpha_{args.alpha}_"
    if args.write and not args.eval:
        os.makedirs(args.record_dir, exist_ok=True)
    if args.save and not args.eval:
        os.makedirs(args.model_dir, exist_ok=True)
    args.record_dir += "/"
    args.model_dir += "/"
    
    if not args.eval:
        print("******************************")
        print("The setting is ", args)
        if not args.write:
            print(" ")
            print("Not going to write files")
            
        else:
            print(" ")
            print("Will write to files")
            
        if args.hit is not None:
            print("Hit Hit Hit !!!")
        print("******************************")
    # idx -> persona sentence
    idx2persona_pool = np.load(args.data_dir + "clean_persona_pool.npy")
    # reshape_data(args, np.load(args.data_dir + "Mapping_idx.npy"))
    # ===== prepare dataset, models and optimizer ==========
    # if args.save and valid_loss < best_valid_loss and valid_acc_1 > best_valid_acc:
    valid_acc_1 = 0
    valid_loss = 0
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    device = torch.device("cuda")
    train_loader, valid_loader = prepare_data_loader(args)
                
    print("Load Data Done")
    bert_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=1608
    )
    bert_model.train()
    model = Model(bert_model)
    
    optimizer = AdamW(model.net.parameters(), lr=args.lr)
    if args.eval or args.hit is not None:
        print("*******************************")
        print("Loading Model ....")
        print("*******************************")
        model.load_state_dict(torch.load(args.load_path + "saved_weights.pt"))
        model.eval()
    model.to(device)
    
    print(
        """
        ######################################################
        finish preparing  !!!!!!!!!!!!!!!!!
        ######################################################"""
    )
    if args.hit is not None:
        hit(model, bert_tokenizer, valid_loader, device, args, K =args.hit)
        exit(0)
    
    best_valid_loss = 100
    best_valid_acc = 0
    #for each epoch
    
    for epoch in range(args.epoch):
        
        print('\n Epoch {:} / {:}'.format(epoch + 1, args.epoch))
        
        #train model
        if not args.eval:
            model = train(model, bert_tokenizer, train_loader, optimizer, args=args, device=device, epoch=epoch)
        # if args.eval:
        valid_loss, valid_acc_1 = evaluate(model, bert_tokenizer, valid_loader, args=args, device=device, epoch=epoch, idx2persona_pool= idx2persona_pool)
        print("Valid loss is ", valid_loss)
        print("Valid acc 1 is ", valid_acc_1)
            # exit(0)
        #evaluate model
        
        #save the best model
        if args.save and valid_loss < best_valid_loss and valid_acc_1 > best_valid_acc:

            with open(args.model_dir +'record.txt', "w") as f:
                f.writelines(f"Valid loss : {valid_loss} \n")
                f.writelines(f"Valid acc1 : {valid_acc_1}")
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), args.model_dir + 'saved_weights.pt')
            # exit(0)
        
    #     count = 0
    #     for input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids in tqdm(
    #         train_loader
    #     ):
    #         # ===== select persona for interlocutor ========
    #         # ===== generate s0 from dataset ===============
    #         print(input_ids.size())
    #         for id in input_ids.squeeze(1):
    #             print("*****************************")
    #             temp = [x for x in tokenizer.decode(id).split() if x not in {'<bos>', '<speaker1>', '<speaker2>', '<pad>', '<|eos|>'}]
    #             temp_sen = ""
    #             for sen in temp:
    #                 temp_sen += (sen + " ")  
    #             print(temp_sen)
    #         exit(0)
    #         interlocutor_persona_enc = []
    #         history_enc = []
    #         score_record = []
    #         for input_i in input_ids:
    #             persona = []
    #             history = []
    #             sen = []
    #             per = True
    #             sen_spk2 = True
    #             #  print(input_i)
    #             for i in range(1, len(input_i[0])):
    #                 if input_i[0][i] != 50258:  # <|eos|>
    #                     if per:
    #                         if input_i[0][i] == 50261:  # <speaker2>
    #                             per = False
    #                         else:
    #                             persona.append(int(input_i[0][i]))
    #                     else:
    #                         if sen_spk2:
    #                             if input_i[0][i] == 50260:
    #                                 sen_spk2 = False
    #                                 history.append(sen)
    #                                 sen = []
    #                             else:
    #                                 sen.append(int(input_i[0][i]))
    #                         else:
    #                             if input_i[0][i] == 50261:
    #                                 sen_spk2 = True
    #                                 history.append(sen)
    #                                 sen = []
    #                             else:
    #                                 sen.append(int(input_i[0][i]))
    #                 else:
    #                     history.append(sen)
    #                     break
    #             interlocutor_persona_enc.append(persona)
    #             history_enc.append(history)
    #         print("****************************************")
    #         print("input id is : \n\t", [tokenizer.decode(id[0]) for id in history_enc])
    #         print("****************************************")
    #         print("mc_token_ids is : \n\t", [tokenizer.decode(idx) for idx in interlocutor_persona_enc])
    #         print("****************************************")
    #         print("lm_labels is : \n\t", lm_labels)
    #         print("****************************************")
    #         print("mc_labels is : \n\t", mc_labels)
    #         print("****************************************")
    #         print("token_type_ids is \n\t", token_type_ids)
    #         yes = True
    #         for per in [tokenizer.decode(idx) for idx in interlocutor_persona_enc]:
    #             if per not in persona_pool:
    #                 yes = False
    #                 print("wrong")
    #                 break
    #         exit(0)
    #         # ===== generate s1 from interlocutor ==========
    #         with torch.no_grad():
    #             response_enc = generate_response(
    #                 interlocutor_persona_enc, history_enc, tokenizer, interlocutor, arg
    #             )
    #             for i in range(args.batch_size):
    #                 history_enc[i].append(response_enc[i])

    #         value_init = get_score([h[-2:] for h in history_enc], tokenizer)

    #         # ===== select persona for s2 ==================
    #         history = [
    #             [tokenizer.decode(s) for s in h[-selector_history:]]
    #             for h in history_enc
    #         ]

    #         if args.select:
    #             # persona_s2, temp_prob, _ , critic_value_0 = select_persona(
    #             #     persona_selector, persona_pool, history, bert_tokenizer, bert_model
    #             # )
    #             # log_prob.extend(temp_prob)

    #             persona_s2 = ppo.select_action(history, bert_tokenizer)
    #         else:
    #             persona_s2, log_prob, _ , critic_value_0 = random.sample(persona, args.batch_size), [
    #                 0 for _ in range(args.batch_size)
    #             ]
    #         # print("Critic value 0 is ", critic_value_0)
    #         # print("Critic value 0 looks like ", np.shape(critic_value_0))
    #         # prob_record.append(log_prob)

    #         # ===== generate s2 ============================
    #         persona_enc = [
    #             tokenizer.encode_plus(p_i, return_attention_mask=False)["input_ids"]
    #             for p_i in persona_s2
    #         ]
    #         with torch.no_grad():
    #             response_enc = generate_response(
    #                 persona_enc, history_enc, tokenizer, model, arg
    #             )
    #             for i in range(args.batch_size):
    #                 history_enc[i].append(response_enc[i])

    #         # ===== generate s3 from interlocutor ==========
    #         with torch.no_grad():
    #             response_enc = generate_response(
    #                 interlocutor_persona_enc, history_enc, tokenizer, interlocutor, arg
    #             )
    #             for i in range(args.batch_size):
    #                 history_enc[i].append(response_enc[i])
    #         ppo.buffer.rewards.append(get_score([h[-2:] for h in history_enc], tokenizer))

    #         # ===== select persona for s4 ==================
    #         history = [
    #             [tokenizer.decode(s) for s in h[-selector_history:]]
    #             for h in history_enc
    #         ]
    #         if not args.fix:
    #             if args.select:
    #                 # persona_s4, temp_prob, _ , critic_value_1 = select_persona(
    #                 #     persona_selector,
    #                 #     persona_pool,
    #                 #     history,
    #                 #     bert_tokenizer,
    #                 #     bert_model,
    #                 # )
    #                 # log_prob.extend(temp_prob)
    #                 persona_s4 = ppo.select_action(history, bert_tokenizer)
    #             else:
    #                 persona_s4, log_prob, _ , critic_value_1 = random.sample(
    #                     persona_pool, args.batch_size
    #                 ), [0 for _ in range(args.batch_size)]
    #         else:
    #             persona_s4, log_prob = persona_s2, [0 for _ in range(args.batch_size)]
    #         # [[v1, v2, ...]]
    #         # print("Critic value 1 is ", np.shape(critic_value_1))
    #         # prob_record.append(log_prob)

    #         # ===== generate s4 ============================
    #         persona_enc = [
    #             tokenizer.encode_plus(p_i, return_attention_mask=False)["input_ids"]
    #             for p_i in persona_s4
    #         ]
    #         with torch.no_grad():
    #             response_enc = generate_response(
    #                 persona_enc, history_enc, tokenizer, model, arg
    #             )
    #             for i in range(args.batch_size):
    #                 history_enc[i].append(response_enc[i])

    #         # ===== generate s5 from interlocutor ==========
    #         with torch.no_grad():
    #             response_enc = generate_response(
    #                 interlocutor_persona_enc, history_enc, tokenizer, interlocutor, arg
    #             )
    #             for i in range(args.batch_size):
    #                 history_enc[i].append(response_enc[i])

    #         # score_record.append(get_score([h[-2:] for h in history_enc], tokenizer))
    #         ppo.buffer.rewards.append(get_score([h[-2:] for h in history_enc], tokenizer))
    #         # for i in range(len(persona_s2)):
    #         #     print("******************************")
    #         #     print("history sentences is \t", ppo.buffer.states[0][i])
    #         #     print("history sentences is \t", ppo.buffer.states[1][i])
    #         #     print("persona_s2 is \t", persona_s2[i])
    #         #     print("reward s2 is ", ppo.buffer.rewards[0][i])
    #         #     print("persona_s4 is \t", persona_s4[i])
    #         #     print("reward s4 is ", ppo.buffer.rewards[1][i])
    #         ppo.update()
    #         count += 1
    #         if count == 1500:
    #             ppo.update(done=True)
    #         # print('value_init\n', value_init)
            # print('prob_record\n', prob_record)
            # print('score_record\n', score_record)

            # rewards_s4 = [score_record[1][i] for i in range(args.batch_size)]
            # rewards_s2 = [score_record[0][i] for i in range(args.batch_size)]

            # rewards_ori = torch.tensor([rewards_s2, rewards_s4], device=arg.device)
            # # torch.Size([2, 16]) [[r2, r2 ....], [r4, r4 ....]]
            # # print("reward_ori is ", np.shape(rewards_ori))
            # rewards = []

            # for r in rewards_ori:
            #     rewards.append((r - r.mean()) / r.std())
            # rewards = (rewards_ori - rewards_ori.mean()) / (rewards_ori.std() + 1e-9)
            # print("rewards is ", np.shape(rewards))
            # print('reward is ', rewards)
            # print("critic 0 is ", np.shape(critic_value_0))
            # print("critic 1 is ", np.shape(critic_value_1))

            # critic_value = [critic_value_0, critic_value_1]
            # print("critic  is ", np.shape(critic_value))
            # print("log_prob is ", log_prob)
            # print("log_prob is ", np.shape(log_prob))

    #         reward is  [tensor([-0.9153,  1.2531, -0.8226,  1.2426,  1.0966, -0.9443,  0.7169,  0.8108,                                                            │·········································
    #     -0.9564, -0.8737,  1.1903,  1.1924, -0.2371, -0.9503, -0.9472, -0.8557],                                                                       │·········································
    #    device='cuda:0'), tensor([-0.5524,  2.0472, -0.5509, -0.5056, -0.4680, -0.3796,  1.7195, -0.5507,                                               │·········································
    #     -0.5527, -0.5483, -0.4664, -0.5019,  0.0245,  2.1970, -0.4153, -0.4964],                                                                       │·········································
    #    device='cuda:0')]  
            # exit(0)
            # loss = 0
            # rewards = rewards[:args.batch_size//2]
            # prob_record = prob_record[:args.batch_size//2]
            # print('rewards\n', rewards)
            # rewards = rewards[:1]
            # prob_record = prob_record[:1]
            # for r, l in zip(rewards, prob_record):
            #     loss -= sum(r * l) 
            #     break

            # rewards_ori = rewards_ori.detach().cpu().numpy()
            # rw_2, rw_4 = np.sum(rewards_ori, axis=1)
            # sc_2, sc_4 = np.sum(score_record, axis=1)
            # sc_ori = np.sum(value_init)

            # # optimizer.zero_grad()
            # # loss /= 2
            # # # persona_selector.train()
            # # loss.backward()
            # # torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
            # # optimizer.step()
            # # scheduler.step()
            # persona_selector.learn(rewards, critic_value, log_prob)
            # i_batch += 1
            # sc += rewards_ori[0].mean()
            # print("loss", loss.item(), "reward:", rewards_ori.mean())
            # if i_batch % 2 == 0:
            #     # torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
            #     persona_selector.optimizer.step()
            #     scheduler.step()
            #     persona_selector.optimizer.zero_grad()
            
        # import matplotlib.pyplot as plt
        # plt.plot(persona_selector.critic_loss_record, label = "critic loss")
        # plt.plot(persona_selector.value_loss_record, label = "value loss")
        # plt.legend(loc = "best")
        # plt.savefig("loss.jpg")
        # exit(0)
            # if i_batch % args.log_step == 0:
            #     niter = i_epoch * len(train_loader) + i_batch
            #     writer.add_scalar("Train/Value_Loss", persona_selector.value_loss, niter)
            #     writer.add_scalar("Train/Critic_Loss", persona_selector.critic_loss, niter)
            #     writer.add_scalar("Train/Score_origin", sc_ori, niter)
            #     writer.add_scalar("Train/Score_s2", sc_2, niter)
            #     writer.add_scalar("Train/Score_s4", sc_4, niter)
            #     writer.add_scalar("Train/Reward_s2", rw_2, niter)
            #     writer.add_scalar("Train/Reward_s4", rw_4, niter)

            # if i_batch % args.print_sample_step == 0:
            #     with open(log_file_path, "a") as fp:
            #         fp.write("\n===== dialogue sample ======================\n")
            #         fp.write(
            #             f"persona_interlocutor :\n{tokenizer.decode(interlocutor_persona_enc[0])}\n"
            #         )
            #         fp.write(f"persona_s2: {persona_s2[0]}\n")
            #         fp.write(f"persona_s4: {persona_s4[0]}\n")
            #         fp.write(f"\nhistory + s1~s5 at {i_epoch} epoch {i_batch} batch\n")
            #         for sen_enc in history_enc[0]:
            #             fp.write(f"{tokenizer.decode(sen_enc)}\n")
            #     print("===== print dialogue sample ==================")
            #     print(
            #         f"\npersona_interlocutor :\n{tokenizer.decode(interlocutor_persona_enc[0])}\n"
            #     )
            #     print("\npersona_s2\n", persona_s2[0])
            #     print("\npersona_s4\n", persona_s4[0])
            #     print(f"\nhistory + s1~s5 at {i_epoch} epoch {i_batch} batch")
            #     for sen_enc in history_enc[0]:
            #         print(tokenizer.decode(sen_enc))
            #     print(sc / args.print_sample_step)
            #     sc = 0

            # if i_batch % args.save_time_step == 0:
            #     torch.save(
            #         persona_selector,
            #         os.path.join(
            #             args.work_space,
            #             args.save_dir,
            #             args.model_name,
            #             f"{i_epoch}_epoch.pkl",
            #         ),
            #     )

        #     prob_record.clear()
        #     score_record.clear()
        # torch.save(
        #     persona_selector,
        #     os.path.join(
        #         args.work_space, args.save_dir, args.model_name, f"{i_epoch}_epoch.pkl"
        #     ),
        # )


if __name__ == "__main__":
    main()
