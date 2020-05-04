import gym
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import *
from collections import deque
from torch.distributions import Categorical
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime

torch.random.manual_seed(1000)

class Network(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256, **kwargs):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_value = nn.Linear(hidden_size, 1)
        self.fc_policy = nn.Linear(hidden_size, action_size)
        self.act = nn.ReLU()

    def forward_value(self, state:torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(state))
        x = self.act(self.fc2(x))
        return self.fc_value(x)

    def forward_policy(self, state:torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(state))
        x = self.act(self.fc2(x))
        return F.softmax(self.fc_policy(x), dim=1)


class Policy:
    def __init__(self, network:Network, **kwargs):
        self.network = network

    def get_action(self, state:torch.Tensor) -> Tuple[float, float]:
        p = self.network.forward_policy(state)
        action_dist = Categorical(p)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action).item()


class Agent:
    def __init__(self, env:gym.Env, epoch=6, lr=1e-4, gamma=0.99, epsilon=0.2, lamda=0.98, **kwargs):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = Network(env.observation_space.shape[-1], env.action_space.n, **kwargs).to(self.device)
        self.optim = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.policy = Policy(self.network, **kwargs)
        self.writer = SummaryWriter(log_dir=f"./runs/ppo-discrete-{datetime.now()}")
        self.global_step, self.train_step = 0, 0
        self.epoch = epoch
        self.epsilon, self.gamma, self.lamda = epsilon, gamma, lamda
        self.memory = []

    def tr(self, x: Iterable) -> torch.Tensor:
        return torch.tensor(np.vstack(x), dtype=torch.float32).to(self.device)

    def get_batch(self) -> Tuple[torch.Tensor, ...]:
        sample = np.array(self.memory).transpose()
        self.memory.clear()
        return tuple(map(self.tr, sample))

    def gae(self, delta):
        adv_lst = []; adv = 0
        for i in reversed(range(delta.shape[0])):
            adv = self.gamma * self.lamda * adv + delta[i]
            adv_lst.append(adv)
        adv_lst.reverse()
        return torch.tensor(adv_lst, dtype=torch.float32).reshape_as(delta)

    def train(self):
        s, a, r, ns, d, p = self.get_batch()
        for eph in range(self.epoch):
            target = r + (1-d) * self.gamma * self.network.forward_value(ns)
            value = self.network.forward_value(s)
            value_loss = F.smooth_l1_loss(self.network.forward_value(s), target)
            delta = target - value
            advantage = self.gae(delta)
            new_p = self.network.forward_policy(s).gather(1, a.long()).log()
            ratio = torch.exp(new_p - p)
            policy_loss = ratio * advantage
            policy_loss_clipped = torch.clamp(ratio, min=1-self.epsilon, max=1+self.epsilon)*advantage
            policy_loss = -torch.min(policy_loss_clipped, policy_loss)[0].mean()
            self.optim.zero_grad()
            (policy_loss + value_loss).backward()
            self.optim.step()
            self.writer.add_scalar("training/loss/policy", policy_loss.item(), self.train_step)
            self.writer.add_scalar("training/loss/policy_abs", policy_loss.abs().item(), self.train_step)
            self.writer.add_scalar("training/loss/value", value_loss.item(), self.train_step)
            self.writer.add_scalar("training/abs_1-ratio", (1-ratio).mean().abs().item(), self.train_step)
            self.writer.add_scalar("training/advantage", advantage.mean().item(), self.train_step)
            self.train_step += 1

    def __call__(self, n_epi=20000, sample_size=25, train=True, objective=450):
        scores = deque(maxlen=10)
        for epi in range(n_epi):
            sc = 0
            s = self.env.reset()
            d = False
            while not d:
                for _ in range(sample_size):
                    a, p = self.policy.get_action(self.tr([s]))
                    ns, r, d, _ = self.env.step(a)
                    self.memory.append((s, [a], [r/100], ns, [d], [p]))
                    sc += r
                    s = ns
                    if d: break
                    self.writer.add_scalar("performance/reward", r, self.global_step)
                    self.writer.add_scalar("action/action", a, self.global_step)
                    self.writer.add_scalar("action/prob", np.exp(p), self.global_step)
                    self.global_step += 1
                if train: self.train()
                else:
                    print("score:", sc)
                    self.env.render()
            self.writer.add_scalar("performance/score", sc, epi)
            scores.append(sc)
            if np.mean(scores) >= objective: break
        self.env.close()

if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    agent = Agent(env=env)
    agent(train=True)
    agent(n_epi=1000, train=False)



