# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain, count

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import (
    AdamW,
    OpenAIGPTDoubleHeadsModel,
    OpenAIGPTTokenizer,
    GPT2DoubleHeadsModel,
    GPT2Tokenizer,
    WEIGHTS_NAME,
    CONFIG_NAME,
)

from utils import get_dataset, make_logdir

SPECIAL_TOKENS = ["<bos>", "<|eos|>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {
    "bos_token": "<bos>",
    "eos_token": "<|eos|>",
    "pad_token": "<pad>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>"],
}
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

logger = logging.getLogger(__file__)


def average_distributed_scalar(scalar, args):
    """Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation."""
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def pad_dataset(dataset, padding=0, pad_left=False):
    """Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler."""
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        if pad_left:
            dataset[name] = [[padding if name != "lm_labels" else -100] * (max_l - len(x)) + x for x in dataset[name]]
        else:
            dataset[name] = [x + [padding if name != "lm_labels" else -100] * (max_l - len(x)) for x in dataset[name]]
    return dataset


def add_special_tokens_(model, tokenizer):
    """Add special tokens to the tokenizer and the model if they have not already been added."""
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)  # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """Build a sequence of input from 3 segments: persona, history and last reply."""
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence) - i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-100] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
    return instance


def get_data_loaders(args, tokenizer):
    """Prepare the dataset for training and evaluation"""
    personachat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)

    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}

    for dataset_name, dataset in personachat.items():
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if args.num_candidates > 0 and dataset_name == "train":
            num_candidates = min(args.num_candidates, num_candidates)
        for dialog in dataset:
            persona = dialog["personality"].copy()
            persona = [persona[0]]
            for _ in range(args.personality_permutations):
                for utterance in dialog["utterances"]:
                    history = utterance["history"][-(2 * args.max_history + 1) :]
                    for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                        lm_labels = bool(j == num_candidates - 1)
                        instance = build_input_from_segments(persona, history, candidate, tokenizer, lm_labels)

                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(input_array)
                    datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                    datasets[dataset_name]["n_candidates"] = num_candidates
                # persona = [persona[0]]
                persona = [persona[-1]] + persona[:-1]  # permuted personalities

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            if input_name != "mc_labels":
                tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed)
    )
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler


def get_data_loaders_persona(args, tokenizer):
    """Prepare the dataset for training and evaluation"""
    personachat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache + "_persona")

    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}

    bos, eos, speaker1, speaker2, pad = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    for dataset_name, dataset in personachat.items():
        for dialog in dataset:
            persona = dialog["personality"].copy()
            for utterance in dialog["utterances"]:
                # skip too short datum
                if len(utterance["history"]) < args.history_turn + 1:
                    continue

                # remain bot response for further comparison
                history = utterance["history"][-(args.history_turn + 1) : -1]

                lens_persona = [len(p) for p in persona]
                lens_history = [len(h) for h in history]

                # remove too long datum
                persona_list = list(chain(*persona))
                if len(persona_list) > args.persona_max_length:
                    continue
                history = list(chain(*history))
                if len(history) > args.history_max_length:
                    continue

                datasets[dataset_name]["persona"].append(persona_list)
                datasets[dataset_name]["history"].append(history)
                datasets[dataset_name]["length_persona"].append(lens_persona)
                datasets[dataset_name]["length_history"].append(lens_history)

                persona = [persona[-1]] + persona[:-1]  # permuted personalities

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    names = ["persona", "history", "length_persona", "length_history"]
    for dataset_name, dataset in datasets.items():
        # padding input
        max_length = [args.persona_max_length, args.history_max_length, 5, args.history_turn]
        padding = [pad, pad, 0, 0]
        for name, max_l, p in zip(names, max_length, padding):
            dataset[name] = [x + [p] * (max_l - len(x)) for x in dataset[name]]

        for name in names:
            tensor = torch.tensor(dataset[name])
            print(dataset_name, name, tensor.shape)
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        shuffle=(not args.distributed),
        worker_init_fn=args.seed,
    )
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler


def get_data_loaders_DialoGPT(args, tokenizer):
    """Prepare the dataset for training and evaluation"""
    dataset_chat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache, encode=False)
    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    for dataset_name, dataset in dataset_chat.items():
        for dialog in dataset:
            for utterance in dialog["utterances"]:
                if len(utterance["history"]) < args.history_turn + 1:
                    continue
                history = utterance["history"][-(args.history_turn + 1) : -1]
                # print("######################")
                # for h in history:
                #     print(h)
                instance = ""
                for h in history[:-1]:
                    instance += h + f" {tokenizer.eos_token} "
                instance += history[-1]
                # print(instance)
                # exit()
                datasets[dataset_name]["history"].append(instance)
    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}

    for dataset_name, dataset in datasets.items():
        names = ["input_ids", "attention_mask"]
        for name in names:
            dataset[name] = []
        count = 0
        for data in dataset["history"]:
            # ignore too long sentences
            if len(data.split()) > args.history_max_length:
                count += 1
                continue

            data_enc = tokenizer.encode_plus(data, max_length=args.history_max_length, padding="max_length", return_tensors="pt")

            # skip some encoding that has strange length
            flag = False
            for name in names:
                if len(data_enc[name][0]) != args.history_max_length:
                    flag = True
                    break
            if flag:
                continue

            for name in names:
                dataset[name].append(data_enc[name].reshape(-1))
        logging.info(f"remove {count} too long data in {dataset_name}")

        for name in names:
            tensor = torch.stack(tuple(dataset[name]), dim=0)
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed)
    )
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler
