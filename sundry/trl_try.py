# imports
import torch
from transformers import GPT2Tokenizer, BertForSequenceClassification, BertTokenizer
from trl_py.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl_py.ppo import PPOTrainer

# get models
# gpt2_model = GPT2HeadWithValueModel.from_pretrained('gpt2')
gpt2_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6732)
gpt2_model_ref = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6732)

# gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained('gpt2')
gpt2_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# initialize trainer
ppo_config = {"batch_size": 1, "forward_batch_size": 1}
ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, **ppo_config)

# encode a query
query_txt = "This morning I went to the "
query_tensor = gpt2_tokenizer.encode(query_txt, return_tensors="pt")

# get model response
response_tensor = respond_to_batch(gpt2_model, query_tensor)
# response_txt = "You went to what ?"
# response_tensor = gpt2_tokenizer.encode(response_txt, return_tensors="pt")
print(response_tensor)
exit(0)
# define a reward for response
# (this could be any reward such as human feedback or output from another model)
reward = torch.tensor([1.0])

# train model with ppo
train_stats = ppo_trainer.step(query_tensor, response_tensor, reward)
# print(train_stats)
print(train_stats.keys())
import json

with open("train_stat.json", "w") as f:

    json.dump(train_stats, f)
