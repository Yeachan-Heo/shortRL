import gym
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import *
from copy import deepcopy
from collections import deque
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime

torch.random.manual_seed(1000)

class PolicyNetwork(nn.Module):
    def __init__(self, state_size:int, action_size:int, hidden_size=32, action_scale=2, **kwargs):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.scale = action_scale
        self.act = nn.ReLU()

    def forward(self, state:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = self.act(self.fc1(state))
        x = self.act(self.fc2(x))
        x = torch.tanh(self.fc3(x))*self.scale
        return x

class QNetwork(PolicyNetwork):
    def __init__(self, state_size:int, action_size:int, hidden_size=32, **kwargs):
        super(QNetwork, self).__init__(state_size, 1, hidden_size, **kwargs)
        self.fc1 = nn.Linear(state_size+action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
        state_action_concat = torch.cat((state, action), -1)
        x = self.act(self.fc1(state_action_concat))
        return self.fc3(self.act(self.fc2(x)))

class Policy:
    def __init__(self, network:PolicyNetwork, scale:float=1.0, scale_decay:float=0.999, scale_min=0.07, **kwargs):
        self.network = network
        self.scale, self.scale_decay, self.scale_min = scale, scale_decay, scale_min

    def get_action(self, state:torch.Tensor):
        action = self.network.forward(state).squeeze(0).cpu().detach().numpy()
        noise = np.random.normal(loc=0, scale=self.scale)
        return action + noise

    def train(self, m):
        if not m: self.scale = self.scale_min = 0

    def decay_scale(self): self.scale = max(self.scale*self.scale_decay, self.scale_min)


class Agent:
    def __init__(self, env:gym.Env, mem_maxlen=20000, minibatch_size=256, n_epoch=20,
                 q_lr=1e-3, policy_lr=1e-4, gamma=0.99, tau=0.1, **kwargs):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNetwork(env.observation_space.shape[-1], env.action_space.shape[-1], **kwargs)\
            .to(self.device)
        self.q_net = QNetwork(env.observation_space.shape[-1], env.action_space.shape[-1], **kwargs)\
            .to(self.device)
        self.target_policy_net, self.target_q_net = deepcopy(self.policy_net), deepcopy(self.q_net)
        self.policy_optim = torch.optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.q_optim = torch.optim.Adam(self.q_net.parameters(), lr=q_lr)
        self.policy = Policy(self.policy_net)
        self.memory = deque(maxlen=mem_maxlen)
        self.minibatch_size, self.n_epoch = minibatch_size, n_epoch
        self.gamma, self.tau = gamma, tau
        self.writer = SummaryWriter(log_dir=f"./runs/ddpg_pendulum-{datetime.now()}")
        self.global_step, self.train_global_step = 0, 0

    def tr(self, x: Iterable) -> torch.Tensor:
        return torch.tensor(np.vstack(x), dtype=torch.float32).to(self.device)

    def sample_from_memory(self) -> Tuple[torch.Tensor, ...]:
        sample = np.array(random.sample(self.memory, self.minibatch_size)).transpose()
        s, ns = self.tr(sample[0]), self.tr(sample[3])
        a, r, d = self.tr(sample[1]), self.tr(sample[2]), self.tr(sample[4])
        return s, a, r, ns, d

    @staticmethod
    def _soft_update(net:nn.Module, target_net:nn.Module, tau:float):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1 - tau) + param.data * tau)

    def soft_update(self):
        self._soft_update(self.policy_net, self.target_policy_net, self.tau)
        self._soft_update(self.q_net, self.target_q_net, self.tau)

    def train(self):
        s, a, r, ns, d = self.sample_from_memory()
        target = r + (d - 1) * self.gamma * self.target_q_net.forward(ns, self.target_policy_net.forward(ns))
        q_loss = F.smooth_l1_loss(self.q_net.forward(s, a), target.detach())
        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()
        self.writer.add_scalar("loss/q_loss", q_loss.item(), self.train_global_step)

        policy_loss = -self.q_net.forward(s, self.policy_net.forward(s)).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        self.writer.add_scalar("loss/policy_loss", policy_loss.item(), self.train_global_step)
        self.train_global_step += 1

    def train_epoch(self):
        for _ in range(self.n_epoch): self.train()

    def __call__(self, n_epi=10000, random_steps=10000, train=True, objective=-350):
        scores = deque(maxlen=10)
        self.policy.train(train)
        for epi in range(n_epi):
            sc = 0
            s = self.env.reset()
            d = False
            while not d:
                a = self.policy.get_action(self.tr([s]))
                ns, r, d, _ = self.env.step(a)
                sc += r
                self.memory.append((s, a, [r], ns, [d]))
                s = ns
                self.writer.add_scalar("action", a.squeeze(), self.global_step)
                self.writer.add_scalar("reward", r, self.global_step)
                self.global_step += 1
            if ((len(self.memory) > random_steps) & train):
                self.train_epoch()
                self.soft_update()
                self.policy.decay_scale()
            if not train :
                self.env.render()
                print("score: ", sc)
            self.writer.add_scalar("score", sc, epi)
            scores.append(sc)
            if np.mean(scores) > sc: break

if __name__ == '__main__':
    agent = Agent(env=gym.make("Pendulum-v0"))
    agent(train=True); agent(n_epi=100, train=False)






