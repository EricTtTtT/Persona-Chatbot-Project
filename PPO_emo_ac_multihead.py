from logging import log

# from collections import OrderedDict
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

print(
    "============================================================================================"
)


# set device to cpu or cuda
device = torch.device("cpu")
persona_num = 4
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

print(
    "============================================================================================"
)


################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.coherence_rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.coherence_rewards[:]
        del self.is_terminals[:]

    def __sizeof__(self) -> int:
        return len(self.actions)

    def size(self):
        return np.shape(self.actions[0])


class ActorCritic(nn.Module):
    def __init__(
        self,
        bert_model,
        persona_pool,
        state_dim=768,
        action_dim=6732,
        action_std_init=0.6,
        sample_persona_num=4,
    ):
        super(ActorCritic, self).__init__()

        # bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6732)
        self.actor = [
            BertForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=len(persona_pool)
            )
            for i in range(sample_persona_num)
        ]
        # critic
        self.critic = [nn.Linear(768, 1) for i in range(sample_persona_num)]

        # self.actor.eval()
        self.persona_pool = persona_pool

        for i in range(sample_persona_num):
            self.actor[i].eval()
            self.critic[i].eval()
            self.actor[i].to(device)
            self.critic[i].to(device)

        # for sample
        self.sample_persona_num = sample_persona_num

    def set_action_std(self, new_action_std):

        print(
            "--------------------------------------------------------------------------------------------"
        )
        print(
            "WARNING : Calling ActorCritic::set_action_std() on discrete action space policy"
        )
        print(
            "--------------------------------------------------------------------------------------------"
        )

    def forward(self):
        raise NotImplementedError

    def act(self, **state):

        actions = []
        log_probs = []
        persona_ids = []
        for actor in self.actor:
            output = actor(**state)
            action_probs = F.softmax(output[0], dim=-1)
            dist = Categorical(action_probs)

            # Replace sample with argmax
            persona_id = dist.sample()

            action_logprob = dist.log_prob(persona_id)
            actions.append(
                [self.persona_pool[id] for id in persona_id.cpu().detach().numpy()]
            )
            log_probs.append(action_logprob)
            persona_ids.append(persona_id)
            # if len(action) :
            #     action = [action[i]+self.persona_pool[id] for id in persona_id.cpu().detach().numpy()]
            # else:
            #     action = [self.persona_pool[id] for id in persona_id.cpu().detach().numpy()]
        return actions, log_probs, persona_ids

        # print("persona id ", persona_id)
        # print(np.shape(persona_id))
        # persona_id = torch.transpose(persona_id, 0, 1)
        # print("persona id ", persona_id)
        # print(np.shape(persona_id))
        # persona_id = torch.argmax(dist)
        # print("Persona id ", persona_id.size())

        # action = [self.persona_pool[id] for id in persona_id.cpu().detach().numpy()]
        # print(actions)
        # print(np.shape(action))
        # print(action_logprob.size())
        # exit(0)
        # return action, torch.transpose(action_logprob, 0, 1), persona_id
        # exit(0)
        # return action.detach(), action_logprob.detach()

    def evaluate(self, idx, action, **state):

        # output is torch.Size([32, 768])
        output = self.actor[idx].bert(**state)

        # State value  is torch.Size([32, 1])
        state_values = self.critic[idx](output[1])

        # Output of classifier  torch.Size([32, 1608])
        output = self.actor[idx].classifier(output[1])

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
        lr_actor=0.000002,
        lr_critic=0.05,
        gamma=0.99,
        K_epochs=3,
        eps_clip=0.5,
        action_std_init=0.3,
        critic_cof=0.5,
        entropy_cof=0.05,
        sample_size=10,
        seed=1000,
        load=False,
        load_path="",
    ):
        # print("output dir is ", self.output_dir)
        # exit(0)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.tokenizer = None
        self.critic_cof = critic_cof
        self.entropy_cof = entropy_cof
        self.buffer = [RolloutBuffer() for i in range(persona_num)]
        self.co_ra = 0.1
        self.load = load
        self.load_checkpoint = "entropy_model.bim"
        print("**********************************************")
        print("lr actor is ", lr_actor)
        print("lr critic is ", lr_critic)
        print("**********************************************")
        # exit(0)
        self.root_dir = "./Emo_PPO_output"
        self.output_dir = f"loss_accum_samplesize_{sample_size}_lra_{lr_actor}_lrc_{lr_critic}_gamma_{gamma}_K_{K_epochs}_eps_{eps_clip}_actionstd_{action_std_init}_cri_{self.critic_cof}_entr_{self.entropy_cof}_seed_{seed}"
        self.output_dir = os.path.join(self.root_dir, self.output_dir)
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.entropy_threshold = 3
        # print("output dir is \n")
        # print(self.output_dir)
        # exit(0)
        if not self.load:
            os.makedirs(self.output_dir, exist_ok=True)

        self.policy = ActorCritic(
            bert_model, persona_pool, state_dim, action_dim, action_std_init
        )
        self.policy_old = ActorCritic(
            bert_model, persona_pool, state_dim, action_dim, action_std_init
        )
        if self.load:
            if load_path is "":
                load_path = os.path.join(self.output_dir, self.load_checkpoint)
            print("load ", load_path)
            self.policy = torch.load(load_path)
            self.policy_old = torch.load(load_path)
        else:
            self.policy_old = ActorCritic(
                bert_model, persona_pool, state_dim, action_dim, action_std_init
            )
            self.policy_old.load_state_dict(self.policy.state_dict())

        parameters = []
        for i in range(persona_num):
            parameters.append(
                {"params": self.policy.actor[i].parameters(), "lr": self.lr_actor}
            )
            parameters.append(
                {"params": self.policy.critic[i].parameters(), "lr": self.lr_critic}
            )
        self.optimizer = torch.optim.Adam(parameters)

        self.optimizer.zero_grad()
        self.loss = 0
        self.MseLoss = nn.MSELoss()

        self.device = "cuda"
        self.loss_record = []
        self.critic_loss_record = []
        self.entropy_record = []
        self.reward_record = []

    def set_action_std(self, new_action_std):

        print(
            "--------------------------------------------------------------------------------------------"
        )
        print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
        print(
            "--------------------------------------------------------------------------------------------"
        )

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print(
            "--------------------------------------------------------------------------------------------"
        )

        print(
            "WARNING : Calling PPO::decay_action_std() on discrete action space policy"
        )
        print(
            "--------------------------------------------------------------------------------------------"
        )

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
        action, action_logprob, persona_id = self.policy_old.act(**encode_input)

        # All of them are one batch [a1, a2, ..., a16], [logp1, logp2, ...., logp16]
        process_action = ["" for i in range(len(action[0]))]
        for i in range(persona_num):
            if not valid:
                self.buffer[i].states.append(history_sentences)
                self.buffer[i].actions.append(persona_id[i])
                self.buffer[i].logprobs.append(action_logprob[i])
            for j in range(len(action[i])):
                process_action[j] += action[i][j]
        return process_action

    def evaluate(self, history_sentences, action):

        log_probs, states, entropy = (
            [None for i in range(len(history_sentences))],
            [None for i in range(len(history_sentences))],
            [None for i in range(len(history_sentences))],
        )
        Total = [[] for i in range(persona_num)]
        Action = [[] for i in range(persona_num)]
        for i in range(len(history_sentences)):
            temp = "[CLS] "
            for s in history_sentences[i]:
                temp += s + " [SEP] "
            Total[i % persona_num].append(temp)
            Action[i % persona_num].append(action[i])

        for i, total in enumerate(Total):
            encode_input = self.tokenizer.batch_encode_plus(
                total,
                add_special_tokens=False,
                pad_to_max_length=True,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            action_logprobs, state_values, dist_entropy = self.policy.evaluate(
                i, torch.tensor(Action[i]).to(device), **encode_input
            )
            for j in range(len(action_logprobs)):
                log_probs[i + 4 * j] = action_logprobs[j]
                states[i + 4 * j] = state_values[j]
                entropy[i + 4 * j] = dist_entropy[j]
        return torch.tensor(log_probs).to(device), torch.tensor(states).to(device), torch.tensor(entropy).to(device)

    def update(self):

        self.optimizer.zero_grad()
        self.optimizer.step()
        # self.loss = 0
        ## Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

    def update_writer(self, turn=2, accum_size=5):
        # turn batch datum in self.buffer to an array
        rewards = []
        coherence_rewards = []
        logprobs = []
        states = []
        actions = []
        # print("reward shape ", self.buffer.rewards)
        # print("reward shape ", np.shape(self.buffer.rewards))
        # print("coherence_reward shape ", self.buffer.coherence_rewards)
        # print("coherence_reward shape ", np.shape(self.buffer.coherence_rewards))
        # print("logprobs shape ", self.buffer.logprobs)
        # print("logprobs shape ", np.shape(self.buffer.logprobs))
        # print("actions.shape ", self.buffer.actions)
        # print("actions.shape ", np.shape(self.buffer.actions))
        # print("states.shape ", self.buffer.states)
        # print("states.shape ", np.shape(self.buffer.states))
        # exit(0)
        for i in range(len(self.buffer[0].rewards[0])):
            for j in range(turn):
                for k in range(persona_num):  # 4
                    rewards.append(self.buffer[k].rewards[j][i])
                    coherence_rewards.append(self.buffer[k].coherence_rewards[j][i])
                    logprobs.append(self.buffer[k].logprobs[j][i])
                    actions.append(self.buffer[k].actions[j][i])
                    states.append(self.buffer[k].states[j][i])

        # print("reward shape ", rewards)
        # print("reward shape ", np.shape(rewards))
        # print("coherence_reward shape ", coherence_rewards)
        # print("coherence_reward shape ", np.shape(coherence_rewards))
        # print("logprobs shape ", logprobs)
        # print("logprobs shape ", np.shape(logprobs))
        # print("actions.shape ", actions)
        # print("actions.shape ", np.shape(actions))
        # print("states.shape ", states)
        # print("states.shape ", np.shape(states))
        # exit(0)
        # Normalizing the rewards
        REWARD = np.mean(rewards)
        coherence_score = np.mean(coherence_rewards)
        # self.reward_record.append(np.mean(rewards))
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        mean = rewards.mean()
        rewards = ((rewards - mean) / (rewards.std() + 1e-7)) + mean

        coherence_rewards = torch.tensor(coherence_rewards, dtype=torch.float32).to(
            self.device
        )
        mean = coherence_rewards.mean()
        coherence_rewards = (
            (coherence_rewards - mean) / (coherence_rewards.std() + 1e-7)
        ) + mean

        rewards = torch.add(coherence_rewards * self.co_ra, rewards) / (self.co_ra + 1)
        # rewards  = rewards.clone().detach()
        # print("reward size : ", rewards.size())
        actions = torch.tensor(actions).to(self.device)
        # logprobs  = logprobs.clone().detach().requires_grad_(True)
        logprobs = torch.tensor(logprobs, requires_grad=True)

        # Optimize policy for K epochs
        # for _ in range(self.K_epochs):
        # Evaluating old actions and values
        new_logprobs, state_values, dist_entropy = self.evaluate(states, actions)
        # new_logprobs,  dist_entropy = self.evaluate(states, actions)
        # match state_values tensor dimensions with rewards tensor
        state_values = torch.squeeze(state_values)

        # Finding the ratio (pi_theta / pi_theta__old)
        ratios = torch.exp(new_logprobs - logprobs.to(self.device))

        # Finding Surrogate Loss
        advantages = rewards - state_values.detach()
        # advantages = rewards
        surr1 = (ratios * advantages).mean()
        surr2 = (
            torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        ).mean()

        # final loss of clipped objective PPO
        # critic_loss = torch.clamp(self.MseLoss(state_values, rewards), 0, 10)
        critic_loss = self.MseLoss(state_values, rewards).mean()
        # print("entropy ", dist_entropy.mean())
        # loss = -torch.min(surr1, surr2) + self.critic_cof*critic_loss + (self.entropy_cof*dist_entropy.mean())
        # if dist_entropy.mean().item() < self.entropy_threshold:
        loss = (
            -torch.min(surr1, surr2)
            + self.critic_cof * critic_loss
            - self.entropy_cof * dist_entropy.mean()
        ) / accum_size
        # else:
        # loss = (-torch.min(surr1, surr2) + self.critic_cof*critic_loss ) / accum_size

        # loss = -torch.min(surr1, surr2) - (self.entropy_cof * dist_entropy.sum())
        # loss = torch.clamp(loss, -1000, 1000)
        if np.random.rand() < 0.05:
            print("surr", -torch.min(surr1, surr2).item())
            print("Advantage is ", advantages.mean().item())
            print("Rewards is ", REWARD)
            print("cohernce score is ", coherence_score)
            print("state value ", state_values.mean().item())
            # print("surr2", surr2)
            # print("advantage : ", advantages)
            print("critic loss : ", critic_loss.item())
            print("dist_entropy.sum()", dist_entropy.mean().item())
            if surr2 < surr1:
                print("!!! surr2 clamp !!!")
            print("loss", loss.item())
        loss.backward()

        # clear buffer
        for idx in range(persona_num):
            self.buffer[idx].clear()
        return (
            loss.item(),
            critic_loss.item(),
            dist_entropy.mean().item(),
            REWARD,
            coherence_score,
        )
        # if np.random.rand() < 0.05:
        #     self.draw()

    def draw(self):

        import matplotlib.pyplot as plt

        # plt.plot(self.critic_loss_record, label = "critic loss")
        plt.plot(self.loss_record, label="loss")
        plt.title("Actor loss")
        plt.legend(loc="best")
        plt.savefig(self.output_dir + "/loss.jpg")
        plt.clf()

        plt.plot(self.critic_loss_record, label="loss")
        plt.title("Critic loss")
        plt.legend(loc="best")
        plt.savefig(self.output_dir + "/critic_loss.jpg")
        plt.clf()

        plt.plot(self.entropy_record, label="entropy")
        plt.title("Entropy")
        plt.legend(loc="best")
        plt.savefig(self.output_dir + "/entropy.jpg")
        plt.clf()

        plt.plot(self.reward_record, label=f"reward")
        # plt.title("Reward")
        plt.legend(loc="best")
        plt.title(
            f"mean = {np.mean(self.reward_record)}, std = {np.std(self.reward_record)}"
        )
        plt.savefig(self.output_dir + "/reward.jpg")
        plt.clf()
        print("Average reward is ", np.mean(self.reward_record))
        print("Std of reward is ", np.std(self.reward_record))
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
