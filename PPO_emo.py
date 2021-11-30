import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import os

# set device to cpu or cuda
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]

    def __sizeof__(self) -> int:
        return len(self.actions)

    def size(self):
        return np.shape(self.actions[0])

class ActorCritic(nn.Module):
    def __init__(self, bert_model, persona_pool, state_dim=768, action_dim=6732, action_std_init=0.6):
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
        action_probs = F.softmax(output[0], dim=-1)
        dist = Categorical(action_probs)

        # Replace sample with argmax
        persona_id = dist.sample()

        action_logprob = dist.log_prob(persona_id)
        action = [self.persona_pool[id] for id in persona_id.cpu().detach().numpy()]

        return action, action_logprob, persona_id
        # return action.detach(), action_logprob.detach()

    def evaluate(self, action, **state):
        # output is torch.Size([32, 768])
        output = self.actor.bert(**state)

        # State value  is torch.Size([32, 1])
        state_values = self.critic(output[1])

        # Output of classifier  torch.Size([32, 1608])
        output = self.actor.classifier(output[1])

        action_probs = F.softmax(output, dim=-1)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, state_values, dist_entropy
        # return action_logprobs,  dist_entropy


class PPO:
    def __init__(
        self,
        persona_pool,
        bert_model,
        state_dim=1608,
        action_dim=6732,
        lr_actor=0.00002,
        lr_critic=0.05,
        gamma=0.99,
        K_epochs=3,
        eps_clip=0.5,
        action_std_init=0.3,
        critic_cof=1.0,
        entropy_cof=0.001,
    ):
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
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.policy = ActorCritic(bert_model, persona_pool, state_dim, action_dim, action_std_init).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": self.lr_actor},
                {"params": self.policy.critic.parameters(), "lr": self.lr_critic},
            ]
        )

        self.policy_old = ActorCritic(bert_model, persona_pool, state_dim, action_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self.device = "cuda"

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
            total, add_special_tokens=False, pad_to_max_length=True, truncation=True, padding=True, return_tensors="pt"
        ).to(self.device)
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
        encode_input = self.tokenizer.batch_encode_plus(
            total, add_special_tokens=False, pad_to_max_length=True, truncation=True, padding=True, return_tensors="pt"
        ).to(self.device)

        return self.policy.evaluate(action, **encode_input)

    def update(self, i_sample, accum_iter, i_batch, step_update, turn=2):
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
        rewards_ori = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards_mean = rewards_ori.mean()
        rewards = ((rewards_ori - rewards_mean) / (rewards_ori.std() + 1e-7)) + rewards_mean

        actions = torch.tensor(actions).to(self.device)
        # logprobs  = logprobs.clone().detach().requires_grad_(True)
        logprobs = torch.tensor(logprobs, requires_grad=True)

        # Optimize policy for K epochs
        loss_sum = 0
        entropy_sum = 0
        critic_loss_sum = 0
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            new_logprobs, state_values, dist_entropy = self.evaluate(states, actions)
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(new_logprobs - logprobs.to(self.device))

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = (ratios * advantages).mean()
            surr2 = (torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages).mean()

            # final loss of clipped objective PPO
            critic_loss = self.MseLoss(state_values, rewards).mean()

            # loss = -torch.min(surr1, surr2) + self.critic_cof*critic_loss + (self.entropy_cof*dist_entropy.mean())
            loss = -torch.min(surr1, surr2) + self.critic_cof * critic_loss

            loss = loss / accum_iter
            loss.backward()

            if (i_sample + 1) == accum_iter and (i_batch + 1) == step_update:
                self.optimizer.step()
                self.optimizer.zero_grad()

            loss_sum += loss.detach().cpu()
            entropy_sum += dist_entropy.detach().cpu()
            critic_loss_sum += critic_loss.detach().cpu()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

        del rewards, logprobs, states, actions
        return {
            "loss": loss_sum / self.K_epochs,
            "entropy": entropy_sum / self.K_epochs,
            "critic_loss": critic_loss_sum / self.K_epochs,
        }

    # def draw(self):

    #     import matplotlib.pyplot as plt

    #     # plt.plot(self.critic_loss_record, label = "critic loss")
    #     plt.plot(self.loss_record, label="loss")
    #     plt.title("Actor loss")
    #     plt.legend(loc="best")
    #     plt.savefig(self.output_dir + "/loss.jpg")
    #     plt.clf()

    #     plt.plot(self.critic_loss_record, label="loss")
    #     plt.title("Critic loss")
    #     plt.legend(loc="best")
    #     plt.savefig(self.output_dir + "/critic_loss.jpg")
    #     plt.clf()

    #     plt.plot(self.entropy_record, label="entropy")
    #     plt.title("Entropy")
    #     plt.legend(loc="best")
    #     plt.savefig(self.output_dir + "/entropy.jpg")
    #     plt.clf()

    #     plt.plot(self.reward_record, label=f"reward")
    #     # plt.title("Reward")
    #     plt.legend(loc="best")
    #     plt.title(f"mean = {np.mean(self.reward_record)}, std = {np.std(self.reward_record)}")
    #     plt.savefig(self.output_dir + "/reward.jpg")
    #     plt.clf()
    #     print("Average reward is ", np.mean(self.reward_record))
    #     print("Std of reward is ", np.std(self.reward_record))
    #     self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
