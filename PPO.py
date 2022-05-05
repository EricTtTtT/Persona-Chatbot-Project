from turtle import forward
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import os

################################## PPO Policy ##################################

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.coherence = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.coherence[:]

    def __sizeof__(self) -> int:
        return len(self.actions)



class ActorCritic(nn.Module):
    def __init__(
        self, bert_model, state_dim=768, action_dim=6732, action_std_init=0.6
    ):
        super(ActorCritic, self).__init__()

        # bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6732)
        self.bert_model = bert_model
        self.actor1 = nn.Linear(state_dim, action_dim)
        self.actor2 = nn.Linear(state_dim, action_dim)
        self.actor3 = nn.Linear(state_dim, action_dim)
        self.actor4 = nn.Linear(state_dim, action_dim)

        self.critic = nn.Linear(state_dim, 1)
        
        self.bert_model.eval()
        self.actor1.eval()
        self.actor2.eval()
        self.actor3.eval()
        self.actor4.eval()

        self.critic.eval()

    def set_action_std(self, new_action_std):

        print("--------------------------------------------------------------------------------------------")
        print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, **state):
        output = self.bert_model.bert(**state)[1]
        probs = [F.softmax(self.actor1(output), dim=-1),
                F.softmax(self.actor2(output), dim=-1),
                F.softmax(self.actor3(output), dim=-1),
                F.softmax(self.actor4(output), dim=-1)]
        probs = torch.stack(probs, dim=1)
        dist = Categorical(probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        logprob = torch.mean(logprob, dim=1)

        return action, logprob

    def evaluate(self, action, **state):

        output = self.bert_model.bert(**state)[1]
        state_values = self.critic(output)

        probs = [F.softmax(self.actor1(output), dim=-1),
                F.softmax(self.actor2(output), dim=-1),
                F.softmax(self.actor3(output), dim=-1),
                F.softmax(self.actor4(output), dim=-1)]
        probs = torch.stack(probs, dim=1)
        dist = Categorical(probs)
        logprob = dist.log_prob(action)
        logprob = torch.mean(logprob, dim=1)
        dist_entropy = dist.entropy()
        dist_entropy = torch.mean(dist_entropy, dim=1)

        return logprob, state_values, dist_entropy


class PPO:
    def __init__(
        self,
        action_dim,
        bert_model,
        state_dim=768,
        lr_actor=0.0002,
        lr_critic=0.5,
        gamma=0.99,
        K_epochs=3,
        eps_clip=0.5,
        action_std_init=0.3,
        critic_cof=0.5,
        entropy_cof=0.2,
        use_entropy_target=False,
        entropy_target=0.4,
        entropy_penalty=1.0,
        coherence_cof=2,
        sample_size=10,
        seed=2,
        load=False,
        load_path="",
    ):
        self.device = "cuda"
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.tokenizer = None
        self.critic_cof = critic_cof
        self.entropy_cof = entropy_cof
        self.use_entropy_target = use_entropy_target
        self.entropy_target = entropy_target
        self.entropy_penalty = entropy_penalty
        self.coherence_cof = coherence_cof
        self.buffer = RolloutBuffer()
        self.load = load
        self.load_checkpoint = "model.bin"
        print("**********************************************")
        print("lr actor is ", lr_actor)
        print("lr critic is ", lr_critic)
        print("**********************************************")

        self.policy = ActorCritic(bert_model, state_dim, action_dim, action_std_init).to(
            self.device
        )
        self.policy_old = ActorCritic(bert_model, state_dim, action_dim, action_std_init).to(
            self.device
        )
        if self.load:
            print("Load ", load_path)
            self.policy = torch.load(load_path)
            self.policy_old = torch.load(load_path)
        else:
            self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(
            [
                # {"params": self.policy.bert_model.parameters(), "lr": lr_critic},
                {"params": self.policy.actor1.parameters(), "lr": lr_actor},
                {"params": self.policy.actor2.parameters(), "lr": lr_actor},
                {"params": self.policy.actor3.parameters(), "lr": lr_actor},
                {"params": self.policy.actor4.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.loss = 0
        self.MseLoss = nn.MSELoss()

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
        encode_input = self.tokenizer.batch_encode_plus(
            total,
            add_special_tokens=False,
            pad_to_max_length=True,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        # with torch.no_grad():
        action, action_logprob = self.policy_old.act(**encode_input)

        # All of them are one batch [a1, a2, ..., a16], [logp1, logp2, ...., logp16]
        if not valid:
            self.buffer.states.append(history_sentences)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

        return action

    def evaluate(self, history_sentences, action):
        total = []
        for i in range(len(history_sentences)):
            temp = "[CLS] "
            for s in history_sentences[i]:
                temp += s + " [SEP] "
            total.append(temp)
        encode_input = self.tokenizer.batch_encode_plus(
            total,
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
        self.policy_old.load_state_dict(self.policy.state_dict())

    def calculate(self, turn=2, accum_size=10):
        # turn batch datum in self.buffer to an array
        rewards = []
        coherence = []
        logprobs = []
        states = []
        actions = []
        for i in range(len(self.buffer.rewards[0])):
            for j in range(turn):
                # not reduce previous rewards
                rewards.append(self.buffer.rewards[j][i])
                coherence.append(self.buffer.coherence[j][i])
                logprobs.append(self.buffer.logprobs[j][i])
                actions.append(self.buffer.actions[j][i])
                states.append(self.buffer.states[j][i])
        
        # print("--------------------------------------------------------------------------------------------")
        # print("rewards is ", rewards)
        # print("coherence is ", coherence)
        # print("logprobs is ", logprobs)
        # print("actions is ", actions)
        # print("states is ", states)
        # print("--------------------------------------------------------------------------------------------")
        # input("Press any key to continue...")

        # Normalizing the rewards
        record_reward = np.mean(rewards)
        record_coherence = np.mean(coherence)

        coherence = torch.tensor(coherence, dtype=torch.float32).to(self.device)
        mean = coherence.mean()
        coherence = (coherence - mean) / (coherence.std() + 1e-7) + mean

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        mean = rewards.mean()
        rewards = (rewards - mean) / (rewards.std() + 1e-7) + mean

        rewards = torch.add(coherence * self.coherence_cof, rewards) / (self.coherence_cof + 1)
        logprobs = torch.tensor(logprobs, requires_grad=True)
        actions = torch.stack(actions, dim=0)

        new_logprobs, state_values, dist_entropy = self.evaluate(states, actions)

        # Finding the ratio (pi_theta / pi_theta__old)
        ratios = torch.exp(new_logprobs - logprobs.to(self.device))
        # Evaluating old actions

        # Finding Surrogate Loss
        advantages = rewards - state_values.detach()
        state_values = torch.squeeze(state_values)

        surr1 = (ratios * advantages).mean()
        surr2 = (torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages).mean()

        critic_loss = self.MseLoss(state_values, rewards).mean()

        # final loss of clipped objective PPO
        entropy_mean = dist_entropy.mean()
        if self.use_entropy_target and entropy_mean <= self.entropy_target:
            loss = (-torch.min(surr1, surr2)) + (self.critic_cof * critic_loss) - self.entropy_penalty * entropy_mean
        else:
            loss = (-torch.min(surr1, surr2)) + (self.critic_cof * critic_loss) - self.entropy_cof * entropy_mean
        loss /= accum_size

        loss.backward()

        # clear buffer
        self.buffer.clear()

        return loss.item(), critic_loss.item(), entropy_mean.item(), record_reward, record_coherence

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
