from logging import log
from typing import OrderedDict
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from transformers import BertForSequenceClassification
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import os 
# from tra

################################## set device ##################################

print("============================================================================================")


# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
    
print("============================================================================================")




################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, bert_model, persona_pool, state_dim = 768, action_dim = 6732, action_std_init = 0.6):
        super(ActorCritic, self).__init__()

        
        # bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6732)
        self.actor = bert_model
        self.actor.eval()
        self.persona_pool = persona_pool
        
        # critic
        self.critic = nn.Linear(768, 1)
        self.critic.eval()
    def set_action_std(self, new_action_std):

        print("--------------------------------------------------------------------------------------------")
        print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")


    def forward(self):
        raise NotImplementedError
    

    def act(self, **state):

        output = self.actor(**state)
        action_probs = F.softmax(output[0], dim = -1)
        dist = Categorical(action_probs)
        # print("Dist ", dist.size())
        # Replace sample with argmax
        persona_id = dist.sample()
        # persona_id = torch.argmax(dist)
        # print("Persona id ", persona_id.size())
        
        action_logprob = dist.log_prob(persona_id)
        action = [self.persona_pool[id] for id in persona_id.cpu().detach().numpy()]
        # exit(0)
        return action, action_logprob, persona_id
        # return action.detach(), action_logprob.detach()
    

    def evaluate(self, action, **state):

        # output is torch.Size([32, 768])
        output = self.actor.bert(**state)
        
        # State value  is torch.Size([32, 1])
        state_values = self.critic(output[1])
        
        # Output of classifier  torch.Size([32, 1608])
        output = self.actor.classifier(output[1])
        
        action_probs = F.softmax(output, dim = -1)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, state_values, dist_entropy
        # return action_logprobs,  dist_entropy


class PPO:
    def __init__(self, persona_pool, bert_model, state_dim = 1608, action_dim = 6732, lr_actor = 0.00002, lr_critic = 0.05,
                 gamma = 0.99, K_epochs = 3, eps_clip  = 0.5, action_std_init=0.3, critic_cof=1.0, entropy_cof = 0.001):
        # print("output dir is ", self.output_dir)
        # exit(0)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.tokenizer = None
        self.critic_cof = critic_cof
        self.entropy_cof = entropy_cof
        self.buffer = RolloutBuffer()
        print("**********************************************")
        print("lr actor is ", lr_actor)
        print("lr critic is ", lr_critic)
        print("**********************************************")
        # exit(0)
        self.root_dir = "./Emo_PPO_output"
        self.output_dir = f"loss_accum_lra_{lr_actor}_lrc_{lr_critic}_gamma_{gamma}_K_{K_epochs}_eps_{eps_clip}_actionstd_{action_std_init}_cri_{self.critic_cof}_entr_{self.entropy_cof}_"
        self.output_dir = os.path.join(self.root_dir, self.output_dir)
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.policy = ActorCritic(bert_model, persona_pool, state_dim, action_dim, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
                    ])

        self.policy_old = ActorCritic(bert_model, persona_pool, state_dim, action_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.loss = 0
        self.MseLoss = nn.MSELoss()

        self.device = "cuda"
        self.loss_record = []
        self.critic_loss_record = []
        self.entropy_record = []
        self.reward_record = []

    def set_action_std(self, new_action_std):
        
        print("--------------------------------------------------------------------------------------------")
        print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")


    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, history_sentences, tokenizer, valid=False):
        if self.tokenizer == None:
            self.tokenizer = tokenizer
        if not valid:
            self.policy_old.eval()

        total = []
        for i in range(len(history_sentences)):
            temp = "[CLS] "
            for s in history_sentences[i]:
                temp += s + " [SEP] "
            total.append(temp)
        encode_input = self.tokenizer.batch_encode_plus(total,
            add_special_tokens=False,
            pad_to_max_length=True,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        # with torch.no_grad():
        action, action_logprob, persona_id = self.policy_old.act(**encode_input)
        
        # All of them are one batch [a1, a2, ..., a16], [logp1, logp2, ...., logp16]
        self.buffer.states.append(history_sentences)
        self.buffer.actions.append(persona_id)
        self.buffer.logprobs.append(action_logprob)

        return action
    def evaluate(self, history_sentences, action):
        
        total = []
        for i in range(len(history_sentences)):
            temp = "[CLS] "
            for s in history_sentences[i]:
                temp += s + " [SEP] "
            total.append(temp)
        encode_input = self.tokenizer.batch_encode_plus(total,
            add_special_tokens=False,
            pad_to_max_length=True,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        return self.policy.evaluate(action, **encode_input)

    def update(self): 
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.loss = 0

    def update_writer(self, writer, i_iter=0, turn=2, step = False, accum_size=5):
        # turn batch datum in self.buffer to an array
        rewards = []
        logprobs = []
        states = []
        actions = []
        for i in range(len(self.buffer.rewards[0])):
            for j in range(turn):
                # not reduce previous rewards
                rewards.append(self.buffer.rewards[j][i])
                logprobs.append(self.buffer.logprobs[j][i])
                actions.append(self.buffer.actions[j][i])
                states.append(self.buffer.states[j][i])
    
        # Normalizing the rewards
        self.reward_record.append(np.mean(rewards))
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        writer.add_scalar("reward", rewards.mean(), i_iter)
        mean = rewards.mean()
        rewards = ((rewards - mean) / (rewards.std() + 1e-7)) + mean

        # rewards  = rewards.clone().detach()
        # print("reward size : ", rewards.size())
        actions  = torch.tensor(actions).to(self.device)
        # logprobs  = logprobs.clone().detach().requires_grad_(True)
        logprobs = torch.tensor(logprobs, requires_grad=True)
        
        # Optimize policy for K epochs
        entropy_sum = 0
        loss_sum = 0
        critic_loss_sum = 0
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            new_logprobs, state_values, dist_entropy = self.evaluate(states, actions)
            # new_logprobs,  dist_entropy = self.evaluate(states, actions)
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(new_logprobs - logprobs.to(self.device))
            
            # Finding Surrogate Loss
            advantages = (rewards - state_values.detach())
            # advantages = rewards
            surr1 = (ratios * advantages).mean()
            surr2 = (torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages).mean()

            # final loss of clipped objective PPO
            # critic_loss = torch.clamp(self.MseLoss(state_values, rewards), 0, 10)
            critic_loss = self.MseLoss(state_values, rewards).mean()
            
            # loss = -torch.min(surr1, surr2) + self.critic_cof*critic_loss + (self.entropy_cof*dist_entropy.mean())
            loss = (-torch.min(surr1, surr2) + self.critic_cof*critic_loss) / accum_size 
            # loss = -torch.min(surr1, surr2) - (self.entropy_cof * dist_entropy.sum())
            # loss = torch.clamp(loss, -1000, 1000)
            if np.random.rand() < 0.05:
                print("surr", -torch.min(surr1, surr2))
                print("Advantage is ", advantages.mean())
                print("Rewards is ", rewards.mean())
                print("state value ", state_values.mean())
                # print("surr2", surr2)
                # print("advantage : ", advantages)
                print("critic loss : ", critic_loss)
                print("dist_entropy.sum()", dist_entropy.mean())
                if surr2 < surr1:
                    print("!!! surr2 clamp !!!")
                print("loss", loss)
            loss.backward()
            if step:
                self.optimizer.step()
                self.optimizer.zero_grad()
            # clip_grad_norm_(self.policy.actor.parameters(), 10)
            # clip_grad_norm_(self.policy.critic.parameters(), 10)
            # take gradient step
            
            loss = loss.detach().cpu()
            dist_entropy = dist_entropy.detach().cpu()
            critic_loss = critic_loss.detach().cpu()
            loss_sum += loss
            entropy_sum += dist_entropy.sum()
            critic_loss_sum += critic_loss
        
        # if loss_sum < 1000 * self.K_epochs:
        self.loss_record.append(loss_sum / self.K_epochs)
        # if critic_loss_sum < 1000 * self.K_epochs:
        self.critic_loss_record.append(critic_loss_sum / self.K_epochs)
        # if entropy_sum < 1000 * self.K_epochs:
        self.entropy_record.append(entropy_sum / self.K_epochs)
        writer.add_scalar("entropy", entropy_sum / self.K_epochs, i_iter)
        writer.add_scalar("loss", loss_sum / self.K_epochs, i_iter)
        writer.add_scalar("critic", critic_loss_sum / self.K_epochs, i_iter)

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        if np.random.rand() < 0.05:
            self.draw()
    def draw(self):
        
        import matplotlib.pyplot as plt
            
            # plt.plot(self.critic_loss_record, label = "critic loss")
        plt.plot(self.loss_record, label = "loss")
        plt.title("Actor loss")
        plt.legend(loc = "best")
        plt.savefig(self.output_dir + "/loss.jpg")
        plt.clf()
        
        plt.plot(self.critic_loss_record, label = "loss")
        plt.title("Critic loss")
        plt.legend(loc = "best")
        plt.savefig(self.output_dir + "/critic_loss.jpg")
        plt.clf()
        
        plt.plot(self.entropy_record, label = "entropy")
        plt.title("Entropy")
        plt.legend(loc = "best")
        plt.savefig(self.output_dir + "/entropy.jpg")
        plt.clf()
        
        plt.plot(self.reward_record, label = f"reward")
        # plt.title("Reward")
        plt.legend(loc = "best")
        plt.title(f"mean = {np.mean(self.reward_record)}, std = {np.std(self.reward_record)}")
        plt.savefig(self.output_dir + "/reward.jpg")
        plt.clf()
        print("Average reward is ", np.mean(self.reward_record))
        print("Std of reward is ", np.std(self.reward_record))
        self.buffer.clear()
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
