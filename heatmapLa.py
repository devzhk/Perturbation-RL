import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from utils import linear_layers
import matplotlib.pyplot as plt

torch.manual_seed(2021)
np.random.seed(2021)

# Hyperparameters
lr_pi = 0.001
lr_q = 0.001
init_alpha = 0.01
gamma = 0.98
batch_size = 32
buffer_limit = 50000
tau = 0.01  # for target network soft update
target_entropy = -1.0  # for automated alpha update
lr_alpha = 0.001  # for automated alpha update

'''
state:
    cos(theta), sin(theta), thetadot
state bound: 
    theta ~ [0, 2 * pi], thetadot ~ [-8.0, 8.0]
'''

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

def transform(theta, thetadot):
    return torch.tensor([np.cos(theta), np.sin(theta), thetadot])

def PDcontrol(x):
    theta = x[0]  # angle_normalize(x[0])
    thetadot = x[1]
    Kp = 5.2
    Kd = 5.2
    u = - Kp * theta - Kd * thetadot
    # 15 * np.sin(x[0] + np.pi)
    return u

class PolicyNet(nn.Module):
    def __init__(self, learning_rate, layers=[3, 128]):
        super(PolicyNet, self).__init__()
        self.fc1 = linear_layers(layers)
        self.fc_mu = nn.Linear(128, 1)
        self.fc_std = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

    def forward(self, x_in):
        x = F.relu(self.fc1(x_in))
        mu = self.fc_mu(x)
        # real_log_prob = torch.tensor([0.0])
        # real_action = mu
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        real_action = torch.tanh(action)
        real_log_prob = log_prob - \
            torch.log(1-torch.tanh(action).pow(2) + 1e-7)
        return 2 * real_action, real_log_prob

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

if __name__ == '__main__':
    ckpt_path = 'sac-3-2000.pt'
    layers = [3, 128, 128]
    # initialize model
    ckpt = torch.load(ckpt_path)
    model = PolicyNet(lr_pi, layers)
    print(layers)
    print('Load weights from %s' % ckpt_path)
    model.load_state_dict(ckpt['policy'])
    sigma = 0.1

    L = 1
    Theta = np.linspace(0, 2*np.pi, L, dtype=np.float32)
    Thetadot = np.linspace(-8, 8, L, dtype=np.float32)
    A_RL = np.zeros([L, L], dtype=np.float32)
    A_model = np.zeros([L, L], dtype=np.float32)

    for i in range(L):
        for j in range(L):
            with torch.no_grad():
                s = transform(Theta[i],Thetadot[j])
                a, _ = model(s)
                A_RL[L-1-i, j] = a.item()
            a = PDcontrol([angle_normalize(Theta[i]), Thetadot[j]])
            A_model[L-1-i, j] = a

            x = torch.tensor([Theta[i],Thetadot[j]], requires_grad=True)
            x.requires_grad = True
            s = torch.tensor([torch.cos(x[0]), torch.sin(x[0]), x[1]])
            output, _ = model(s)
            print('s: ', s.requires_grad)
            output.backward()
            print('Output: ', output)
            print(x.grad.data)

    # plt.subplot(1, 2, 1)
    plt.pcolor(Thetadot, Theta, A_RL, cmap='Reds_r')
    plt.colorbar()
    plt.title('Action Heatmap (SAC)', fontsize=16)
    plt.xlabel(r'$\dot{\theta}$', fontsize=16)
    plt.ylabel(r'$\theta$', fontsize=16)

    # plt.subplot(1, 2, 2)
    # plt.pcolor(Thetadot, Theta, A_model, cmap='Reds_r')
    # plt.colorbar()
    # plt.title('Action Heatmap (PD control)', fontsize=16)
    # plt.xlabel(r'$\dot{\theta}$', fontsize=16)
    # plt.ylabel(r'$\theta$', fontsize=16)

    plt.show()

    # perturbation = torch.rand(2) * sigma
    # a, log_prob = model(transform(theta, thetadot))
    # a_p, _ = model(transform(theta + perturbation[0], thetadot + perturbation[1]))
    # La = torch.norm(a-a_p, p=2) / torch.norm(perturbation, p=2)
    # max_La = La.item() if La.item() > max_La else max_La








