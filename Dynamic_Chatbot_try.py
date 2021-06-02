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
# from collections import defaultdict
from pprint import pformat

import numpy as np
import torch
import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer, BertModel, BertTokenizer
from torch.nn.utils.rnn import pad_sequence
# from utils import get_dataset, download_pretrained_model
from train_try import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_, get_data_loaders

from Engaging_classifier import analyze_engagement
from Persona_Selector_try import prepare_persona_selector, select_persona
from tensorboardX import SummaryWriter
writer = SummaryWriter('runs')

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits

def top_filtering_bt(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
    """
    # batch support!
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
       # print(values.shape)
        min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
        logits = torch.where(logits < min_values, 
                             torch.ones_like(logits, dtype=logits.dtype) * -float('Inf'), 
                             logits)
    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        sorted_logits = sorted_logits.masked_fill_(sorted_indices_to_remove, filter_value)
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)
    
    return logits

def sample_sequence(personality, history, tokenizer, model, arg, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(arg.max_length):
        instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)
        input_ids = torch.tensor(instance["input_ids"], device = arg.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device = arg.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / arg.temperature
        logits = top_filtering(logits, top_k=arg.top_k, top_p=arg.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if arg.no_sample else torch.multinomial(probs, 1)
        if i < arg.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

def generate_response_bt(personality, history, tokenizer, model, arg, current_output=None):
    ret = []
    for persona_i, history_i in zip(personality, history):
        ret.append(sample_sequence([persona_i], history_i, tokenizer, model, arg))
    return ret
    
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    bos, eos, speaker1, speaker2, pad = special_tokens_ids
    if current_output is None:
        current_output = [[] for _ in range(arg.train_batch_size)]
    
    sequence_bt = [[[bos] + persona_i] + history_i + [[eos]] for persona_i, history_i in zip(personality, history)]
    sequence_bt = [[seq[0]] + [[speaker2 if (len(seq)-i) % 2 else speaker1] + s for i, s in enumerate(seq[1:])] for seq in sequence_bt]
    token_type_ids_bt = [[speaker2 if i % 2 else speaker1 for i, s in enumerate(seq) for _ in s] for seq in sequence_bt]
    sequence_bt = [list(chain(*seq)) for seq in sequence_bt]
    reply_position = [len(seq) for seq in sequence_bt]
    max_len = max(reply_position)
    for seq, tok in zip(sequence_bt, token_type_ids_bt):
        padding = [pad for _ in range(max_len - len(seq))]
        seq.extend(padding)
        tok.extend(padding)

    sequence_bt = torch.tensor(sequence_bt, device=arg.device)
    token_type_ids_bt = torch.tensor(token_type_ids_bt, device=arg.device)

    # print('sequence_bt.size()', sequence_bt.size())
    # print('token_type_ids_bt.size()', token_type_ids_bt.size())

    for i_word in range(arg.max_length):
        # prev_input, past = model()
        logits = model(sequence_bt, token_type_ids=token_type_ids_bt)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits / arg.temperature
        logits = top_filtering(logits, top_k=arg.top_k, top_p=arg.top_p)
        print('logits.size()', logits.size()) # (8, 128, 50262)
        probs = F.softmax(logits, dim=-1)
        prev = [torch.topk(prob_i, 1)[1] if arg.no_sample else torch.multinomial(prob_i, 1) for prob_i in probs]
        for prev_i, prob_i in zip(prev, probs):
            if i_word < arg.min_length and prev_i.item() in special_tokens_ids:
                while prev_i.item() in special_tokens_ids:
                    if prob_i.max().item() == 1:
                        warnings.warn("Warning: model generating special token with probability 1.")
                        break  # avoid infinitely looping over special token
                    prev_i = torch.multinomial(prob_i, num_samples=1)

        if i_word == 0:
            for i_bt in range(len(reply_position)):
                current_output[i_bt].append(prev[i_bt].item())
            exit()
            continue

        flag = True
        for i_bt in range(len(reply_position)):
            if prev[i_bt].item() != eos:
                flag = False
                current_output[i_bt].append(prev[i_bt].item())
        for prev_i in prev:
            if prev_i.item() != eos:
                flag = False
                break
        if flag:
            break
        

        current_output = [out_i + prev_i.item() for out_i, prev_i in zip(current_output, prev)]
        print(current_output)
        sequence_bt = [torch.cat((seq_i[:pos_i], torch.tensor([prev_i.item()], device=arg.device), seq_i[pos_i:]), 1) for seq_i, pos_i, prev_i in zip(sequence_bt, reply_position, prev)]
        # print(sequence_bt)
        reply_position = [pos_i + 1 for pos_i in reply_position]
        # print(reply_position)

    return current_output

def prepare_chatbot(check_point, bt=4, root='.'):
    class ARG:
        def __init__(self):
            self.dataset_path = os.path.join(root, 'data/personachat_self_original.json')
            self.dataset_cache = os.path.join(root, 'data/dataset_cache')
            self.max_history = 2
            self.num_candidates = 1
            self.device = "cuda"
            self.no_sample = False
            self.max_length = 20
            self.min_length = 1
            self.seed = 2
            self.temperature = 1
            self.top_k = 0
            self.top_p= 0
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
    tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) #if args.model == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
    tokenizer = tokenizer_class.from_pretrained(check_point)
    model = model_class.from_pretrained(check_point)
    interlocutor = model_class.from_pretrained(check_point)
    model.to(arg.device).eval()
    interlocutor.to(arg.device).eval()

    add_special_tokens_(model, tokenizer)
    add_special_tokens_(interlocutor, tokenizer)

    return model, interlocutor, tokenizer, arg


def main():
    parser = ArgumentParser()
    parser.add_argument("--work_space", type = str, default=".")
    parser.add_argument("--model_checkpoint", type = str, default="model/gpt2_persona_model/")
    parser.add_argument("--batch_size", type = int, default=32)
    parser.add_argument("--epoch", type = int, default=1)
    parser.add_argument("--lr", type = float, default=1e-5)
    parser.add_argument("--save_dir", type = str, default="model/")
    parser.add_argument("--dir_name", type = str, default="bt32_lr1e-5") # persona selector model folder
    parser.add_argument("--load_model_path", type = str, default='')
    parser.add_argument("--log_file", type = str, default="record_bt32_lr1e-5.txt")
    parser.add_argument("--log_step", type = int, default=2)
    parser.add_argument("--print_sample_step", type = int, default=20)
    parser.add_argument("--save_time_step", type = int, default=100)
    parser.add_argument("--select", type = bool, default=True)
    parser.add_argument("--fix", type = bool, default=True)
    
    args = parser.parse_args()
    os.makedirs(os.path.join(args.work_space, args.save_dir, args.dir_name), exist_ok=True)

    #===== prepare dataset, models and optimizer ==========
    model, interlocutor, tokenizer, arg = prepare_chatbot(os.path.join(args.work_space, args.model_checkpoint), bt=args.batch_size)
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(arg, tokenizer)
    del val_loader, train_sampler, valid_sampler
    print('\n\nlen(train_loader): ', len(train_loader), '\n\n')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()
    
    persona_selector, persona_pool = prepare_persona_selector(load_path=args.load_model_path)
    optimizer = torch.optim.Adam(persona_selector.id_selector.parameters(), lr = args.lr)
    
    selector_history = 5

    print("""
        ######################################################
        finish preparing  !!!!!!!!!!!!!!!!!
        ######################################################""")
    
    for i_epoch in range(args.epoch):
        i_batch = 0
        for input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids in train_loader:
            #===== select persona for interlocutor ========
            #===== generate s0 from dataset ===============            
            interlocutor_persona_enc = []
            history_enc = []
            score = 0

            for input_i in input_ids:
                persona = []
                history = []
                sen = []
                per = True
                sen_spk2 = True
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
            # print('######################')
            # print('persona')
            # print(tokenizer.decode(interlocutor_persona_enc[0]))
            # print('######################')
            # print('history init')
            # for sen_enc in history_enc[0]:
            #     print(tokenizer.decode(sen_enc))
    
            #===== generate s1 from interlocutor ==========
            with torch.no_grad():
                response_enc = generate_response_bt(interlocutor_persona_enc, history_enc, tokenizer, interlocutor, arg)
                for i in range(args.batch_size):
                    history_enc[i].append(response_enc[i])

            #===== select persona for s2 ==================
            history = [[tokenizer.decode(s) for s in h[-selector_history:]] for h in history_enc ]
            
            if args.select:
                persona_s2, log_prob = select_persona(persona_selector, persona_pool, history, bert_tokenizer, bert_model)
            else:
                persona_s2 = random.sample(persona, args.batch_size)

            #===== generate s2 ============================
            persona_enc = [tokenizer.encode_plus(p_i, return_attention_mask=False)['input_ids'] for p_i in persona_s2]
            with torch.no_grad():
                response_enc = generate_response_bt(persona_enc, history_enc, tokenizer, model, arg)
                for i in range(args.batch_size):
                    history_enc[i].append(response_enc[i])

            #===== generate s3 from interlocutor ==========
            with torch.no_grad():    
                response_enc = generate_response_bt(interlocutor_persona_enc, history_enc, tokenizer, interlocutor, arg)
                for i in range(args.batch_size):
                    history_enc[i].append(response_enc[i])

            #===== select persona for s4 ==================
            history = [[tokenizer.decode(s) for s in h[-selector_history:]] for h in history_enc ]
            if not args.fix:
                if args.select:
                    persona_s4, log_prob = select_persona(persona_selector, persona_pool, history, bert_tokenizer, bert_model)
                else:
                    persona_s4 = random.sample(persona_pool, args.batch_size)
            else:
                persona_s4 = persona_s2

            #===== generate s4 ============================
            persona_enc = [tokenizer.encode_plus(p_i, return_attention_mask=False)['input_ids'] for p_i in persona_s4]
            with torch.no_grad():
                response_enc = generate_response_bt(persona_enc, history_enc, tokenizer, model, arg)
                for i in range(args.batch_size):
                    history_enc[i].append(response_enc[i])

            #===== generate s5 from interlocutor ==========
            with torch.no_grad():
                response_enc = generate_response_bt(interlocutor_persona_enc, history_enc, tokenizer, interlocutor, arg)
                for i in range(args.batch_size):
                    history_enc[i].append(response_enc[i])

            # print('\nhistory + s0, s1, s2, s3, s4, s5')
            # for sen_enc in history_enc[0]:
            #     print(tokenizer.decode(sen_enc))

            s4 = []
            s5 = []
            for history_enc_i in history_enc:
                s4.append(tokenizer.decode(history_enc_i[-2]))
                s5.append(tokenizer.decode(history_enc_i[-1]))

            score_ori = analyze_engagement(s4, s5)
            score_ori = torch.tensor(score_ori, device=arg.device)
            score = (score_ori - score_ori.mean()) / score_ori.std()
            loss = 0

            loss += sum(score * log_prob)
            score = list(score.detach().cpu().numpy())

            optimizer.zero_grad()
            persona_selector.train()
            loss.backward()
            optimizer.step()

            i_batch += 1
            if i_batch % args.log_step == 0:
                niter = i_epoch*len(train_loader)+i_batch
                print('loss', loss.item())
                writer.add_scalar('Train/Loss', loss, niter)
                writer.add_scalar('Train/Score', score_ori.mean(), niter)
    
            if i_batch % args.print_sample_step == 0:
                with open(args.log_file, 'a') as fp:
                    fp.write("\n===== dialogue sample ======================\n")
                    fp.write(f"persona_s2: {persona_s2[0]}\n")
                    fp.write(f"persona_s4: {persona_s4[0]}\n")
                    fp.write(f"\nhistory + s1~s5 at {i_epoch} epoch {i_batch} batch\n")
                    for sen_enc in history_enc[0]:
                        fp.write(f"{tokenizer.decode(sen_enc)}\n")
                print("===== print dialogue sample ==================")
                print('\npersona_s2\n', persona_s2[0])
                print('\npersona_s4\n', persona_s4[0])
                print(f"\nhistory + s1~s5 at {i_epoch} epoch {i_batch} batch")
                for sen_enc in history_enc[0]:
                    print(tokenizer.decode(sen_enc))

            if i_batch % args.save_time_step == 0:
                torch.save(persona_selector, os.path.join(args.work_space, args.save_dir, args.dir_name, f"{i_epoch}_epoch.pkl"))
        torch.save(persona_selector, os.path.join(args.work_space, args.save_dir, args.dir_name, f"{i_epoch}_epoch.pkl"))
if __name__ == "__main__":
    main()    
    