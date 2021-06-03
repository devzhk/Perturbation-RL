from ppo import PPO

import math
import numpy as np
import torch

torch.manual_seed(2021)
np.random.seed(2021)


def MC(model, in_dim, out_dim, sample_num=10000, sigma=0.01):
    state = torch.rand((sample_num, in_dim))
    perturbation = torch.rand((sample_num, in_dim)) * sigma
    mu, std = model.pi(state)
    mu_p, std_p = model.pi(state + perturbation)
