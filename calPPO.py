from ppo import PPO

import math
import numpy as np
import torch


def MC(model, in_dim, sample_num=10000, sigma=0.01):
    state = torch.rand((sample_num, in_dim))
    perturbation = torch.rand((sample_num, in_dim)) * sigma
    mu, std = model.pi(state)
    mu_p, std_p = model.pi(state + perturbation)
    diff = torch.norm(mu - mu_p, p=2, dim=1)
    norm = torch.norm(perturbation, p=2, dim=1)
    La = diff / norm
    maxLa = torch.max(La).item()
    return maxLa


if __name__ == '__main__':
    sigma = 0.01
    epoch = 1002
    sample_num = 1_000_000
    ckpt_path = 'checkpoints/ppo-%d.pt' % epoch
    state_dim = 2
    action_dim = 1
    model = PPO(in_dim=state_dim, out_dim=action_dim)

    ckpt = torch.load(ckpt_path)
    print('Load weights from %s' % ckpt_path)
    model.load_state_dict(ckpt['policy'])
    maxLa = MC(model, state_dim, sample_num, sigma)
    print(f'MC: {maxLa}')