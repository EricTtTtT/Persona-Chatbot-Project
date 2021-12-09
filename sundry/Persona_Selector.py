from itertools import accumulate
import json
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from transformers import BertTokenizer, BertModel

device = torch.device("cuda:0")
selector_history = 5

# ===== Pytorch Model definition ======================
class PersonaSelector(nn.Module):
    def __init__(self, bert_model, lr):
        super(PersonaSelector, self).__init__()

        # not using dropout for RL training stability
        # Input dim : 768
        # Output dim : 6732
        self.value_net = bert_model

        self.critic_net = nn.Sequential(nn.Linear(768, 192), nn.Linear(192, 48), nn.Linear(48, 1))
        self.value_optim = torch.optim.Adam(self.value_net.parameters(), lr=lr * 3)
        self.critic_optim = torch.optim.Adam(self.critic_net.parameters(), lr=lr * 3)

        self.gamma = 0.9
        self.value_loss = 0
        self.critic_loss = 0

        self.value_count = 0
        self.critic_count = 0

        self.critic_loss_record = []
        self.value_loss_record = []

    def forward(self, **encode_input):
        output = self.value_net(**encode_input, output_hidden_states=True)
        last_layer_output = output[1][0].mean(1)
        # print("size of last layer ", np.shape(last_layer_output))
        # exit(0)
        critic_value = self.critic_net(last_layer_output)
        # print("critic is ", critic_value)
        # print("critic looks like ", np.shape(critic_value))

        prob = F.softmax(output[0], dim=-1)
        distribution = Categorical(prob)
        # print("distri is ", distribution)
        entropy = distribution.entropy()
        persona_id = distribution.sample()
        log_prob = distribution.log_prob(persona_id)
        # exit(0)
        return persona_id.cpu().detach().numpy(), log_prob, entropy, critic_value

    def learn(self, rewards, critic_value, log_prob):
        # pass
        accumulate_reward = []
        Advantage = []
        for i in range(len(rewards[0])):
            accumulate_reward.append(rewards[0][i] + self.gamma * rewards[1][i])
            Advantage.append(accumulate_reward[-1] - critic_value[0][i])
            accumulate_reward.append(rewards[1][i])
            Advantage.append(accumulate_reward[-1] - critic_value[1][i])

        # print("Accumulate is ", accumulate_reward)
        # print("Accumulate is ", np.shape(accumulate_reward))
        # print("Advantage is ", Advantage)
        # print("Advantage is ", np.shape(Advantage))
        Advantage = torch.tensor(Advantage, requires_grad=True)
        log_prob = torch.tensor(log_prob, requires_grad=True)
        accumulate_reward = torch.tensor(accumulate_reward, requires_grad=True)
        # To change the shape of critic_value
        new_critic_value = []
        for c in critic_value:
            new_critic_value.extend(c)
        critic_value = torch.tensor(new_critic_value, requires_grad=True)
        self.value_loss -= (Advantage * log_prob).sum()
        self.value_loss_record.append(self.value_loss.item())
        loss_func = nn.SmoothL1Loss()
        # print("accumulate is ", accumulate_reward)
        # print("accumulate is ", accumulate_reward.size())
        # print("critic is ", critic_value)
        # print("critic is ", critic_value.size())
        self.critic_loss = loss_func(accumulate_reward, critic_value)
        self.critic_loss_record.append(self.critic_loss.item())
        print("Value loss is ", self.value_loss)
        print("Critic loss is ", self.critic_loss)

        self.value_optim.zero_grad()
        self.value_loss.backward()
        self.value_optim.step()

        self.critic_optim.zero_grad()
        self.critic_loss.backward()
        self.critic_optim.step()
        self.value_loss = 0
        self.critic_loss = 0
        # exit(0)


def prepare_persona_selector():
    # ==========Training Prepare===========================
    # persona_selector = PersonaSelector(bert_model, lr)
    # if load_path != "":
    #     persona_selector = torch.load(load_path)
    # persona_selector.cuda()
    # persona_selector.train()
    # persona_selector.id_selector.train()

    # ==========setting IO=================================
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

    return persona_pool


def select_persona(persona_selector, persona_pool, history_sentences, tokenizer, model, valid=False):
    if not valid:
        persona_selector.train()
    persona_selector.to(device)
    model.to(device)
    model.train()
    # print(history_sentences)
    total = []
    for i in range(len(history_sentences)):
        temp = "[CLS] "
        for s in history_sentences[i]:
            temp += s + " [SEP] "
        total.append(temp)
    encode_input = tokenizer(total, add_special_tokens=False, truncation=True, padding=True, return_tensors="pt").to(device)
    persona_id, log_prob, entropy, critic_value = persona_selector(**encode_input)
    # output = model(**encode_input)
    # persona_id, log_prob, entropy, critic_value = persona_selector(output[1])
    selected_persona = [persona_pool[id] for id in persona_id]
    #     #history_sentences[i] = [x + ' [SEP] ' for x in hi]
    # # print("history_sentences is \n :", history_sentences)
    # # print("np.shape(history_sentences) is", np.shape(history_sentences))
    # encoded_input = []
    # for sentences in history_sentences:
    #     dialogue_enc = []
    #     pad_len = selector_history - len(sentences)
    #     # if pad_len > 0:
    #     #     warnings.warn(f"Warning: history size {len(sentences)} is shorter than {selector_history}.")
    #     #     for i in range(pad_len):
    #     #         dialogue_enc.append([0 for _ in range(40)])
    #     for sen in sentences:
    #         enc = tokenizer.encode_plus(
    #             text = sen,
    #             add_special_tokens=True,
    #             max_length = 40,
    #             pad_to_max_length = True,
    #             return_attention_mask = False
    #         )
    #         dialogue_enc.append(enc['input_ids'])
    #     encoded_input.append(dialogue_enc)

    # encoded_input = torch.tensor(encoded_input, device = device)
    # ps_input_arr = []
    # for dialogue in encoded_input:
    #     logits = model(dialogue)
    #     #log = model(input_ids=dialogue, return_dict=True)
    #     if isinstance(logits, tuple):
    #         # logits = logits[0]
    #         temp = logits[0]

    #     ps_input = torch.mean(temp, 1).squeeze(1)
    #    # print(torch.equal(ps_input, logits[1]))
    #     print(ps_input.shape)
    #     ps_input = torch.cat(tuple(ps_input), 0)
    #     print(ps_input.shape)
    #     ps_input_arr.append(ps_input)

    # ps_input_arr = torch.stack(tuple(ps_input_arr)).to(torch.device(device))
    # persona_id, log_prob, entropy = persona_selector(ps_input_arr)
    # selected_persona = [persona_pool[id] for id in persona_id]
    # print('selected persona:\n', selected_persona)

    return selected_persona, log_prob, entropy, critic_value


# ===== testing functions =============================
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
# persona_selector, persona_pool = prepare_persona_selector()

# history_sentences = [['i love to listen to frank sinatra.', 'how is life going for you?', 'it is going well. i am not poor, my wife hates me.', "123", "456"],
#                     ['i am a older lady.', "how old are you? i'm 32", "i'm a 32 year old male.", "123", "456"],
#                     ['i love to eat cheese.', 'my girlfriend just broke up with me.', 'i love to read. i can not afford a television', "123", "456"],
#                     ['i like to cook stews.', 'i love making french fries', 'i like to shop.', "123", "456"],
#                     ['i love to listen to frank sinatra.', 'how is life going for you?', 'it is going well. i am not poor, my wife hates me.', "123", "456"],
#                     ['i am a older lady.', "how old are you? i'm 32", "i'm a 32 year old male.", "123", "456"],
#                     ['i love to eat cheese.', 'my girlfriend just broke up with me.', 'i love to read. i can not afford a television', "123", "456"],
#                     ['i like to cook stews.', 'i love making french fries'] ]

# selected_persona, log_prob = select_persona(persona_selector, persona_pool, history_sentences, tokenizer, model)
# print(selected_persona)
