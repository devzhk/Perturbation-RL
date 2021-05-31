import math
import gym
import torch
import torch.optim as optim

from sac import PolicyNet

sigma = math.sqrt(0.5)
PI = math.pi
lr_pi = 0.0005
epoch = 5000
ckpt_path = 'checkpoints/ep-%d.pt' % epoch

max_speed = 8.0

'''
state:
    cos(theta), sin(theta), thetadot
state bound: 
    theta ~ [0, 4* pi], thetadot ~ [-8.0, 8.0]
'''


def transform(theta, thetadot):
    return torch.cat([torch.cos(theta), torch.sin(theta), thetadot])


def train(model, iter_num=100):
    # initialization
    theta = torch.rand(1) * 4 * PI
    theta.requires_grad = True
    thetadot = torch.rand(1) * 2 * max_speed - max_speed
    thetadot.requires_grad = True

    perturbation = torch.rand(2)
    perturbation.requires_grad = True

    optimizer = optim.SGD(params=[theta, thetadot, perturbation], lr=10)

    for i in range(iter_num):
        a, log_prob = model(transform(theta + perturbation[0], thetadot + perturbation[1]))
        negLa = - torch.norm(a, p=2)

        optimizer.zero_grad()
        negLa.backward()
        optimizer.step()
        theta.clamp(min=0, max=4 * PI)
        thetadot.clamp(min=max_speed, max=max_speed)
        if i % 2 == 0:
            print('La: {}'.format(-negLa.item()))


def MCMC(model, sample_num=10000):
    max_La = 0.0
    for i in range(sample_num):
        theta = torch.rand(1) * 2 * PI
        thetadot = torch.rand(1) * 2 * max_speed - max_speed
        perturbation = torch.rand(2) * sigma
        a, log_prob = model(transform(theta + perturbation[0], thetadot + perturbation[1]))
        La = torch.norm(a, p=2)
        max_La = La if La > max_La else max_La
    return max_La


if __name__ == '__main__':
    model = PolicyNet(lr_pi)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['policy'])
    train(model, iter_num=100)
    print(MCMC(model, sample_num=10000))




