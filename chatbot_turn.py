# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
import logging
import warnings
from itertools import chain
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import torch
import tensorflow as tf

from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer, BertForSequenceClassification
from torch.nn.utils.rnn import pad_sequence

from train import SPECIAL_TOKENS, add_special_tokens_, get_data_loaders_persona
import wandb

from PPO_emo import PPO

from Engaging_classifier import analyze_engagement

# from GoEmotions_pytorch.model import EmoBertForMultiLabelClassification
# from GoEmotions_pytorch.multilabel_pipeline import MultiLabelPipeline

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

    mask = [[1] * len(seq) for seq in sequence]
    sequence = torch.LongTensor(
        tf.keras.preprocessing.sequence.pad_sequences([torch.LongTensor(s) for s in sequence], value=tokenizer.encode("<pad>")[0])
    ).to(arg.device)
    token_type_ids = torch.LongTensor(
        tf.keras.preprocessing.sequence.pad_sequences([torch.LongTensor(s) for s in token_type_ids], value=speaker1)
    ).to(arg.device)
    mask = torch.LongTensor(tf.keras.preprocessing.sequence.pad_sequences([torch.LongTensor(s) for s in mask], value=0)).to(
        arg.device
    )
    position_ids = mask.long().cumsum(-1) - 1  # + prev_input.shape[1]
    position_ids.masked_fill_(mask == 0, 1)

    _, past = model(sequence, attention_mask=mask, token_type_ids=token_type_ids, position_ids=position_ids)

    mask_append = torch.LongTensor([[1] for _ in range(len(history))]).to(arg.device)
    token_tp = torch.LongTensor([[speaker2] if len(x) % 2 else [speaker1] for x in history]).to(arg.device)
    prev = torch.LongTensor([[speaker2] if len(x) % 2 else [speaker1] for x in history]).to(arg.device)

    temp_sen = [[] for i in range(len(history))]
    for i_word in range(arg.max_length):
        mask = torch.cat((mask, mask_append), 1)
        position_ids = mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(mask == 0, 1)
        position_ids = position_ids[:, -1].unsqueeze(-1)

        logits, past = model(prev, token_type_ids=token_tp, past=past, attention_mask=mask, position_ids=position_ids)
        logits = logits.squeeze(0).squeeze(1)
        # logits = top_filtering(logits, top_k=arg.top_k, top_p=arg.top_p)
        probs = torch.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1]
        # prev = torch.multinomial(probs, num_samples=1)
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


def prepare_chatbot(check_point, bt=8, seed=2021):
    class ARG:
        def __init__(self):
            self.dataset_path = "data/personachat_self_original.json"
            self.dataset_cache = "data/cache_persona_3his"
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

    arg = ARG()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)

    random.seed(arg.seed)
    torch.random.manual_seed(arg.seed)
    torch.cuda.manual_seed(arg.seed)

    logger.info("Get pretrained model and tokenizer")
    tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel)
    # if args.model == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)

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


def validation(data_loader, model, interlocutor, tokenizer, bert_tokenizer, ppo, arg, args):
    print("Validation")
    reward_turn_sum = [0 for i in range(args.turn)]
    count = 0
    with torch.no_grad():
        for inter_persona_ori, history_ori, len_p, len_h in tqdm(data_loader):
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

            score_record = []
            persona_bot_record = []
            for i_turn in range(args.turn):
                # get chatbot persona
                history = [[tokenizer.decode(s) for s in h] for h in history_enc]
                ppo.buffer.states.extend(history.copy())
                persona_bot = ppo.select_action(history, bert_tokenizer)
                persona_bot_enc = [tokenizer.encode(p) for p in persona_bot]
                persona_bot_record.append(persona_bot)

                # generate one turn response
                with torch.no_grad():
                    # chatbots
                    response_enc = generate_response(persona_bot_enc, history_enc, tokenizer, model, arg)
                    history_enc = [h + [r] for h, r in zip(history_enc, response_enc)]

                    # interlocutor
                    response_enc = generate_response(inter_persona_enc, history_enc, tokenizer, interlocutor, arg)
                    history_enc = [h + [r] for h, r in zip(history_enc, response_enc)]
                score = get_score([h[-2:] for h in history_enc], tokenizer)
                score_record.append(sum(score) / len(score))

            for i_turn in range(args.turn):
                reward_turn_sum[i_turn] += score_record[i_turn]
            count += 1
        ppo.buffer.clear()
        ppo.buffer.sum_clear()

    for i_turn in range(args.turn):
        reward_turn_sum[i_turn] /= count

    # [reward_mean_turn_0, reward_mean_turn_1, ...]
    return reward_turn_sum


def main():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--gpt2_persona_checkpoint", type=str, default="model/gpt2_persona_model/")
    parser.add_argument("--save_dir", type=str, default="model/")
    parser.add_argument("--model_name", type=str, default="engaging")

    # hyper parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--lr_actor", type=float, default=2e-5)
    parser.add_argument("--lr_critic", type=float, default=2e-4)
    parser.add_argument("--turn", type=int, default=1)
    parser.add_argument("--sample_iter", type=int, default=10)

    # ppo
    parser.add_argument("--K_epochs", type=int, default=3)
    parser.add_argument("--weight_critic", type=float, default=1.0)
    parser.add_argument("--weight_entropy", type=float, default=0.2)
    parser.add_argument("--use_threshold_entropy", type=bool, default=False)
    parser.add_argument("--threshold_entropy", type=float, default=100.0)

    # steps
    parser.add_argument("--step_sample", type=int, default=50)
    parser.add_argument("--step_save", type=int, default=1000)
    parser.add_argument("--step_update", type=int, default=1)
    parser.add_argument("--step_valid", type=int, default=5)

    args = parser.parse_args()
    args.model_save_folder = os.path.join(args.root, args.save_dir, args.model_name)
    os.makedirs(args.model_save_folder, exist_ok=True)
    args.sample_file = os.path.join(args.root, f"sample/{args.model_name}.txt")

    # ===== prepare dataset, models and optimizer ==========
    model, interlocutor, tokenizer, arg = prepare_chatbot(
        os.path.join(args.root, args.gpt2_persona_checkpoint), bt=args.batch_size
    )
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders_persona(arg, tokenizer)
    del train_sampler, valid_sampler

    # persona_pool = remove_duplicate_persona()
    persona_pool = np.load("./clean_persona.npy")
    print("shape of persona_pool", np.shape(persona_pool))

    bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(persona_pool))
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # ====== emotion score ==========
    # emo_tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
    # emo_model = EmoBertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")
    # emo_tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-group")
    # emo_model = EmoBertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-group")
    # goemotions = MultiLabelPipeline(model=emo_model, tokenizer=emo_tokenizer, threshold=0.3)

    ppo = PPO(
        persona_pool=persona_pool,
        bert_model=bert_model,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        K_epochs=args.K_epochs,
        critic_cof=args.weight_critic,
        entropy_cof=args.weight_entropy,
        threshold_entropy=args.threshold_entropy,
    )

    # wandb.init(project="persona_chatbot", entity="erictien")
    wandb.init(
        project="engaging-chatbot",
        entity="persona_chatbot_ntuee",
        config={
            "batch_size": args.batch_size,
            "epoch": args.epoch,
            "lr_actor": args.lr_actor,
            "lr_critic": args.lr_critic,
            "turn": args.turn,
            "sample_iter": args.sample_iter,
            "step_update": args.step_update,
            "K_epochs": args.K_epochs,
            "weight_critic": args.weight_critic,
            "weight_entropy": args.weight_entropy,
            "use_threshold_entropy": args.use_threshold_entropy,
        },
    )

    print(
        """
        ######################################################
        finish preparing  !!!!!!!!!!!!!!!!!
        ######################################################
    """
    )

    i_batch = 0
    for i_epoch in range(args.epoch):
        loss_sum = 0
        loss_critic_sum = 0
        reward_sum = 0
        entropy_sum = 0
        for inter_persona_ori, history_ori, len_p, len_h in tqdm(train_loader):
            if i_batch % args.step_valid == 0:
                turn_rewards = validation(val_loader, model, interlocutor, tokenizer, bert_tokenizer, ppo, arg, args)
                rewards_record = {}
                for i_turn in range(args.turn):
                    rewards_record[f"valid_reward_turn_{i_turn}"] = turn_rewards[i_turn]
                rewards_record["valid_reward"] = sum(turn_rewards) / args.turn
                wandb.log(rewards_record, step=i_batch)

            # recover inter_persona and history from padded datum
            inter_persona_enc = []
            for p_ori, lens in zip(inter_persona_ori, len_p):
                l = sum(lens)
                inter_persona_enc.append(p_ori[:l].tolist())
            history_enc_ori = []
            for h_ori, lens in zip(history_ori, len_h):
                tmp = []
                j = 0
                for l in lens:
                    if l > 0:
                        tmp.append((h_ori[j : j + l]).tolist())
                        j += l
                history_enc_ori.append(tmp)

            # TODO: for K_epoch
            for i_k in range(args.K_epochs):
                for i_sample in range(args.sample_iter):
                    persona_bot_record = []
                    history_enc = history_enc_ori.copy()

                    for i_turn in range(args.turn):
                        # get chatbot persona
                        history = [[tokenizer.decode(s) for s in h] for h in history_enc]
                        ppo.buffer.states.extend(history.copy())
                        persona_bot = ppo.select_action(history, bert_tokenizer)
                        persona_bot_enc = [tokenizer.encode(p) for p in persona_bot]
                        persona_bot_record.append(persona_bot)

                        # generate one turn response
                        with torch.no_grad():
                            # chatbots
                            response_enc = generate_response(persona_bot_enc, history_enc, tokenizer, model, arg)
                            history_enc = [h + [r] for h, r in zip(history_enc, response_enc)]

                            # interlocutor
                            response_enc = generate_response(inter_persona_enc, history_enc, tokenizer, interlocutor, arg)
                            history_enc = [h + [r] for h, r in zip(history_enc, response_enc)]
                        # score_text = [tokenizer.decode(h[-1]) for h in history_enc]
                        # ppo.buffer.rewards.extend(goemotions.get_positive_score(score_text))
                        score = get_score([h[-2:] for h in history_enc], tokenizer)
                        ppo.buffer.rewards.extend(score)

                    ppo.calculate(use_threshold=args.use_threshold_entropy)

                record = ppo.step(sample_iter=args.sample_iter)
                loss_sum += record["loss"]
                loss_critic_sum += record["loss_critic"]
                reward_sum += record["reward"]
                entropy_sum += record["entropy"]

            ppo.update()

            # if (i_batch+1) % args.step_update == 0:
            wandb.log(
                {
                    "loss": loss_sum / args.step_update / args.K_epochs,
                    "loss_critic": loss_critic_sum / args.step_update / args.K_epochs,
                    "reward": reward_sum / args.step_update / args.K_epochs,
                    "entropy": entropy_sum / args.step_update / args.K_epochs,
                },
                step=i_batch,
            )
            loss_sum = 0
            loss_critic_sum = 0
            reward_sum = 0
            entropy_sum = 0

            # TODO: different sample
            if i_batch % args.step_sample == 0:
                sample_str = "\n#########################\n"
                for j in range(args.batch_size):
                    sample_str += "\n#########################\n"
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
                torch.save(ppo.policy, os.path.join(args.model_save_folder, "model.bin"))

            i_batch += 1

    wandb.save(os.path.join(args.model_save_folder, "model.bin"))


if __name__ == "__main__":
    main()
