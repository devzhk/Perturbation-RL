import math
import numpy as np

import torch
import torch.optim as optim
from tqdm import tqdm
from sacPD import PolicyNet


torch.manual_seed(2021)
np.random.seed(2021)

sigma = 0.01
PI = math.pi
lr_pi = 0.0005

max_speed = 8.0

'''
state:
    cos(theta), sin(theta), thetadot
state bound: 
    theta ~ [0, 2 * pi], thetadot ~ [-8.0, 8.0]
'''


def transform(theta, thetadot):
    return torch.cat([torch.cos(theta), torch.sin(theta), thetadot], dim=1)


def train(model, iter_num=100):
    # initialization
    theta = torch.rand(1) * 4 * PI
    theta.requires_grad = True
    thetadot = torch.rand(1) * 2 * max_speed - max_speed
    thetadot.requires_grad = True

    perturbation = torch.rand(2)
    perturbation.requires_grad = True
    max_La = 0.0
    optimizer = optim.Adam(params=[theta, thetadot, perturbation], betas=(0.0, 0.999), lr=1e-4)
    # milestones = [10000, 20000, 30000]
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    pbar = tqdm(range(iter_num), dynamic_ncols=True, smoothing=0.01)
    for i in pbar:
        # scheduler.step()
        a, log_prob = model(transform(theta, thetadot))
        a_p, _ = model(transform(theta + perturbation[0], thetadot + perturbation[1]))
        negLa = - torch.norm(a - a_p, p=2) / torch.norm(perturbation, p=2)

        optimizer.zero_grad()
        negLa.backward()
        optimizer.step()
        theta.clamp(min=0, max=4 * PI)
        thetadot.clamp(min=max_speed, max=max_speed)
        max_La = -negLa.item() if -negLa.item() > max_La else max_La
        pbar.set_description(
            (
                f'La: {max_La}'
            )
        )
    return max_La


def MCMC(model, sample_num=10000):
    max_La = 0.0
    pbar = tqdm(range(sample_num), dynamic_ncols=True, smoothing=0.01)
    for i in pbar:
        theta = torch.rand(1) * 2 * PI
        thetadot = torch.rand(1) * 2 * max_speed - max_speed
        perturbation = torch.rand(2) * sigma
        a, log_prob = model(transform(theta, thetadot))
        a_p, _ = model(transform(theta + perturbation[0], thetadot + perturbation[1]))
        La = torch.norm(a-a_p, p=2) / torch.norm(perturbation, p=2)
        max_La = La.item() if La.item() > max_La else max_La
        pbar.set_description(
            (
                f'La: {max_La}'
            )
        )
    return max_La


def MC(model, sample_num=10000, sigma=0.01):
    theta = torch.rand((sample_num, 1)) * 2 * PI
    thetadot = torch.rand((sample_num, 1)) * 2 * max_speed - max_speed

    perturbation = torch.rand((sample_num, 2)) * sigma
    mu, std = model(transform(theta, thetadot))
    mu_p, std_p = model(transform(theta + perturbation[:, 0:1], thetadot + perturbation[:, 1:]))
    diff = torch.norm(mu - mu_p, p=2, dim=1)
    norm = torch.norm(perturbation, p=2, dim=1)
    La = diff / norm
    maxLa = torch.max(La).item()
    return maxLa


def estimate(model, func, num_trial=5, iter_num=50000):
    La_list = []
    for i in range(num_trial):
        la = func(model, iter_num)
        print('Trial :{}, La: {}'.format(i, la))
        La_list.append(la)
    Las = np.array(La_list)
    max = Las.max()
    mean = Las.mean()
    return mean, max


if __name__ == '__main__':
    epoch = 2000  # options: 1000, 0

    ckpt_path = 'checkpoints/sac-4-%d.pt' % epoch
    layers = [3, 128, 128, 128]
    model = PolicyNet(lr_pi, layers)
    print(layers)

    #  load model weights
    ckpt = torch.load(ckpt_path)
    print('Load weights from %s' % ckpt_path)
    model.load_state_dict(ckpt['policy'])

    La = MC(model, sample_num=100000)
    print(f'La : {La}')
    # print('Estimation via optimization')
    # mean, std = estimate(model, train, 5, 50000)
    # print('Mean: {}, std: {}'.format(mean, std))

    # print('Estimation via random sampling')
    # mean, max = estimate(model, MCMC, 5, 100000)
    # print('Mean: {}, std: {}'.format(mean, max))
    # print(MCMC(model, sample_num=100000))




