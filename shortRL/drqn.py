import gym
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import *
from copy import deepcopy
from datetime import datetime
from collections import deque
from torch.utils.tensorboard.writer import SummaryWriter

class Network(nn.Module):
    # q network in discrete action space
    def __init__(self, state_size:int, action_size:int, hidden_size:int=32, *args, **kwargs):
        super(Network, self).__init__()
        self.fc1:nn.Module = nn.LSTM(state_size, hidden_size)
        self.fc2:nn.Module = nn.LSTM(hidden_size, hidden_size)
        self.fc3:nn.Module = nn.Linear(hidden_size, action_size)
        self.act:Callable = F.tanh

    # feed-forward process
    def forward(self, state:torch.Tensor) -> torch.Tensor:
        state = state.unsqueeze(1)
        x = self.act(self.fc1(state)[0])
        x = self.act(self.fc2(x)[0])
        x = self.act(self.fc3(x))
        x = x.squeeze(1)
        return x


class Policy:
    # epsilon-greedy policy
    def __init__(self, network:Network, eps_dec:float=0.9995, eps_min:float=0.01, *args, **kwargs):
        self.eps = 1
        self.network = network
        self.eps_dec = eps_dec
        self.eps_min = eps_min

    def epsilon_decay(self):
        self.eps = max(self.eps_min, self.eps*self.eps_dec)

    def train(self, m):
        if not m: self.eps_min = self.eps = 0

    def get_action(self, state:torch.Tensor) -> Tuple[int, float, float]:
        q = self.network.forward(state) # q값 구하기
        if self.eps < np.random.uniform():
            action = q.argmax(1).item() # 그리디 액션 (exploitation)
        else:
            action = np.random.randint(0, q.shape[-1]) # 랜덤 액션 (exploration)
        max_q = q.max(1)[0].item() # max q 값
        avg_q = q.mean(1).item() # 평균 q 값
        return action, avg_q, max_q


class Agent:
    def __init__(self, env: gym.Env, mem_maxlen=10000, minibatch_size=256,
                 random_step=5000, lr=0.0005, gamma=0.99, *args, **kwargs):
        self.env = env
        self.writer = SummaryWriter(log_dir=f"./runs/dqn-{datetime.now()}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = Network(state_size=self.env.observation_space.shape[0],
                               action_size=self.env.action_space.n, *args, **kwargs).to(self.device)
        self.target_network = deepcopy(self.network)
        self.opt = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.policy = Policy(network=self.network, *args, **kwargs)
        self.memory = deque(maxlen=mem_maxlen)
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.random_step = random_step
        self.total_step = 0
        self.do_eval = False

    def tr(self, x: Iterable) -> torch.Tensor:
        return torch.tensor(np.vstack(x), dtype=torch.float32).to(self.device)

    def sample_from_memory(self) -> Tuple[torch.Tensor, ...]:
        sample = np.array(random.sample(self.memory, self.minibatch_size)).transpose()
        s, ns = self.tr(sample[0]), self.tr(sample[3])
        a, r, d = self.tr(sample[1]), self.tr(sample[2]), self.tr(sample[4])
        return s, a, r, ns, d

    def train(self):
        if self.total_step < self.random_step:
            return
        s, a, r, ns, d = self.sample_from_memory()
        q = self.network.forward(s).gather(1, a.long())
        nq = self.target_network.forward(ns).max(1)[0].unsqueeze(1)
        target = r + (d - 1) * self.gamma * nq
        loss = F.smooth_l1_loss(q, target.detach())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.writer.add_scalar("loss", loss.item(), self.total_step)
        self.policy.epsilon_decay()

    def __call__(self, n_epi=10000, target_update_interval: int = 1000, goal=180, train=True):
        sc_lst = deque(maxlen=5)
        self.policy.train(train)
        for epi in range(n_epi):
            s = self.env.reset()
            sc = 0
            d = False
            while not d:
                a, avg_q, max_q = self.policy.get_action(self.tr([s]))
                ns, r, d, _ = self.env.step(a)
                sc += r
                self.memory.append((s, [a], [r], ns, [d]))
                if not train: self.train()
                if train: self.env.render()
                self.writer.add_scalar("action/action", a, self.total_step)
                self.writer.add_scalar("q/avg_q", avg_q, self.total_step)
                self.writer.add_scalar("q/max_q", max_q, self.total_step)
                self.writer.add_scalar("action/advantage", max_q - avg_q, self.total_step)
                self.writer.add_scalar("epsilon", self.policy.eps, self.total_step)
                self.total_step += 1
                if self.total_step % target_update_interval == 0:
                    self.target_network.load_state_dict(self.network.state_dict())
                s = ns
            self.writer.add_scalar("score", sc, epi)
            sc_lst.append(sc)
            if ((np.mean(sc_lst) >= goal) & train): break


if __name__ == '__main__':
    agent = Agent(gym.make("CartPole-v0"))
    agent()
    agent(n_epi=100, train=False)
