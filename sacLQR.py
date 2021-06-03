from envs.LQR import LQR
from utils import get_AB

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import collections
import random
import os
from tqdm import tqdm

try:
    import wandb
except:
    wandb = None

# Hyperparameters
lr_pi = 0.0005
lr_q = 0.001
init_alpha = 0.01
gamma = 1.0
batch_size = 32
buffer_limit = 50000
tau = 0.01  # for target network soft update
target_entropy = -1.0  # for automated alpha update
lr_alpha = 0.001  # for automated alpha update


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
            torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)


class PolicyNet(nn.Module):
    def __init__(self, learning_rate, in_dim, out_dim=1):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc_mu = nn.Linear(128, out_dim)
        self.fc_std = nn.Linear(128, out_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

    def forward(self, x_in):
        x = F.relu(self.fc1(x_in))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x)) + 1e-6
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        real_action = torch.tanh(action)
        real_log_prob = log_prob - \
            torch.log(1-torch.tanh(action).pow(2) + 1e-7)
        return 10 * real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s, a), q2(s, a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -min_q - entropy  # for gradient ascent
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() *
                       (log_prob + target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()


class QNet(nn.Module):
    def __init__(self, learning_rate, in_dim=2, out_dim=1):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(in_dim, 64)
        self.fc_a = nn.Linear(out_dim, 64)
        self.fc_cat = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

    def train_net(self, target, mini_batch):
        s, a, r, s_prime, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(s, a), target)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(
                param_target.data * (1.0 - tau) + param.data * tau)


def calc_target(pi, q1, q2, mini_batch):
    s, a, r, s_prime, done = mini_batch

    with torch.no_grad():
        a_prime, log_prob = pi(s_prime)
        entropy = -pi.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(s_prime, a_prime), q2(s_prime, a_prime)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        target = r + gamma * done * (min_q + entropy)

    return target


def main(log=False):
    T = 100
    scale = 10_000
    dt = 1.0
    state_dim = 2
    if log and wandb:
        wandb.init(project='RL-LQR',
                   entity='hzzheng',
                   config={
                       'T': T,
                       'Cost scale': scale,
                       'dt': dt,
                       'state dim': state_dim,
                       'gamma': gamma,
                   })

    action_dim = 1
    A, B = get_AB(state_dim, action_dim, dt)
    # A, B = random_AB(state_dim, action_dim)
    sigma = 0.1
    W = sigma * np.eye(state_dim)
    Q = np.eye(state_dim) * 10.0
    R = np.eye(action_dim)

    env = LQR(A, B, Q, R, W, state_dim)

    memory = ReplayBuffer()
    q1, q2, q1_target, q2_target = \
        QNet(lr_q, state_dim, action_dim), QNet(lr_q, state_dim, action_dim), \
        QNet(lr_q, state_dim, action_dim), QNet(lr_q, state_dim, action_dim)
    pi = PolicyNet(lr_pi, in_dim=state_dim, out_dim=action_dim)

    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    score = 0.0
    print_interval = 10

    save_dir = 'checkpoints'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pbar = tqdm(range(600), dynamic_ncols=True, smoothing=0.1)
    for n_epi in pbar:
        s = env.reset(factor=1.0)
        done = False
        for j in range(T):
            a, log_prob = pi(torch.from_numpy(s).float())
            s_prime, r, done, info = env.step(a.detach().numpy())
            memory.put((s, a.item(), r/1.0, s_prime, done))
            score += r / T / scale
            s = s_prime

        if memory.size() > 1000:
            for i in range(20):
                mini_batch = memory.sample(batch_size)
                td_target = calc_target(pi, q1_target, q2_target, mini_batch)
                q1.train_net(td_target, mini_batch)
                q2.train_net(td_target, mini_batch)
                entropy = pi.train_net(q1, q2, mini_batch)
                q1.soft_update(q1_target)
                q2.soft_update(q2_target)
        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f} alpha:{:.4f}".format(
                n_epi, score/print_interval, pi.log_alpha.exp()))
            if log and wandb:
                wandb.log(
                    {
                        'Avg score': score/print_interval,
                        'Alpha': pi.log_alpha.exp()
                    }
                )
            score = 0.0

        if n_epi % 1000 == 0:
            state_dict = pi.state_dict()
            torch.save({'policy': state_dict}, save_dir +
                       '/ep-{}.pt'.format(n_epi))


if __name__ == '__main__':
    main()
