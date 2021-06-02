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

#===== Pytorch Model definition ======================
class PersonaSelector(nn.Module):
    def __init__(self):
        super(PersonaSelector, self).__init__()

        # not using dropout for RL training stability
        self.id_selector = nn.Sequential(
            nn.Linear(768*selector_history, 6732)
        )

    def forward(self, x):
        x = self.id_selector(x)
        prob = F.log_softmax(x, dim=-1)
        distribution = Categorical(prob)
        entropy = distribution.entropy()
        persona_id = distribution.sample()
        log_prob = distribution.log_prob(persona_id)
        return persona_id.cpu().detach().numpy(), log_prob, entropy

def prepare_persona_selector(load_path=''):
    #==========Training Prepare===========================
    persona_selector = PersonaSelector()
    if load_path != '':
        persona_selector = torch.load(load_path)
    persona_selector.cuda()
    persona_selector.train()
    persona_selector.id_selector.train()
    
    #==========setting IO=================================
    persona_data_path = './data/personachat_self_original.json'
    persona_data_file = open(persona_data_path)
    persona_data = json.load(persona_data_file)

    #==========read persona sentences=====================
    data_type_list = ['train', 'valid']
    persona_set = set()
    for data_type in data_type_list:
        count = 0
        for i in range(len(persona_data[data_type])):
            count += len(persona_data[data_type][i]['personality'])
            for i_sentence in persona_data[data_type][i]['personality']:
                persona_set.add(i_sentence)
        print(data_type, 'data size: ', count)
    print('total # of persona: ', len(persona_set))
    persona_pool = sorted(list(persona_set))

    return persona_selector, persona_pool

def select_persona(persona_selector, persona_pool, history_sentences, tokenizer, model, valid=False):
    if not valid:
        persona_selector.train()
    persona_selector.to(device)
    model.to(device)
    model.eval()
    # print("history_sentences is \n :", history_sentences)
    # print("np.shape(history_sentences) is", np.shape(history_sentences))
    encoded_input = []
    for sentences in history_sentences:
        dialogue_enc = []
        pad_len = selector_history - len(sentences)
        if pad_len > 0:
            warnings.warn(f"Warning: history size {len(sentences)} is shorter than {selector_history}.")
            for i in range(pad_len):
                dialogue_enc.append([0 for _ in range(32)])

        for sen in sentences:
            enc = tokenizer.encode_plus(
                text = sen,
                add_special_tokens=True,
                max_length = 32,
                pad_to_max_length = True,
                return_attention_mask = False
            )
            dialogue_enc.append(enc['input_ids'])
        encoded_input.append(dialogue_enc)
    
    encoded_input = torch.tensor(encoded_input, device = device)
    ps_input_arr = []
    for dialogue in encoded_input:
        logits = model(dialogue)
        if isinstance(logits, tuple):
            # logits = logits[0]
            temp = logits[0]
        ps_input = torch.mean(temp, 1).squeeze(1)
        ps_input = torch.cat(tuple(ps_input), 0)
        ps_input_arr.append(ps_input)
        
    ps_input_arr = torch.stack(tuple(ps_input_arr)).to(torch.device(device))
    persona_id, log_prob, entropy = persona_selector(ps_input_arr)
    selected_persona = [persona_pool[id] for id in persona_id]
    # print('selected persona:\n', selected_persona)
   
    return selected_persona, log_prob


#===== testing functions =============================
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