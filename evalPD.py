import math
import numpy as np

import gym
import torch
from sacPD import PolicyNet
from tqdm import tqdm


torch.manual_seed(2021)
np.random.seed(2021)

sigma = 0.1
PI = math.pi
lr_pi = 0.0005
epoch = 1000  # options: 1000, 0

ckpt_path = 'checkpoints/ep-%d.pt' % epoch

max_speed = 8.0

'''
state:
    cos(theta), sin(theta), thetadot
state bound: 
    theta ~ [0, 2 * pi], thetadot ~ [-8.0, 8.0]
'''


def add_noise(s, sigma=0.01):
    '''
    s: [cos, sin, thetadot]
    Add noise to s
    Let x denote theta, y denote perturbation
    cos(x+y) = cos(x)cos(y) - sin(x)sin(y)
    sin(x+y) = sin(x)cos(y) + sin(y)cos(x)
    :param sigma:
    :return:
    '''

    z = np.random.rand(2) * sigma
    sinz = np.sin(z[0])
    cosz = np.cos(z[0])

    cos = cosz * s[0] - sinz * s[1]
    sin = sinz * s[0] + cosz * s[1]
    new_thetadot = s[2] + z[1]
    return np.array([cos, sin, new_thetadot])


def transform(theta, thetadot):
    return torch.cat([torch.cos(theta), torch.sin(theta), thetadot])


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


def PDcontrol(x):
    theta = x[0]  # angle_normalize(x[0])
    thetadot = x[1]
    Kp = 5.2
    Kd = 5.2
    u = - Kp * theta - Kd * thetadot
    # 15 * np.sin(x[0] + np.pi)
    return u


def main(sigma=0.01):
    print(f'Noise :{sigma}')
    env = gym.make('Pendulum-v0')
    score = 0.0
    print_interval = 20
    sample_num = 1000
    pbar = tqdm(range(sample_num), dynamic_ncols=True, smoothing=0.01)
    scores = []
    for n_epi in pbar:
        s = env.reset()
        state = env.state
        done = False

        while not done:
            a = PDcontrol(state)
            s_prime, r, done, info = env.step([a])
            # env.render()
            state = env.state + np.random.rand(2) * sigma
            score += r
        scores.append(score)
        pbar.set_description(
            (
                f'Round : {n_epi} Score :{score}'
            )
        )
        score = 0.0
    env.close()
    print(f'Mean of score: {np.mean(scores)}')


def eval_sac(model, sigma=0.01):
    print(f'Noise: {sigma}')
    # create environment
    env = gym.make('Pendulum-v0')
    score = 0.0

    sample_num = 500
    pbar = tqdm(range(sample_num), dynamic_ncols=True, smoothing=0.01)
    scores = []
    for n_epi in pbar:
        s = env.reset()
        done = False

        while not done:
            a, log_prob = model(torch.from_numpy(s).float())
            s_prime, r, done, info = env.step([a.item()])
            # env.render()
            score += r
            s = add_noise(s_prime, sigma)
        scores.append(score)
        pbar.set_description(
            (
                f'Round : {n_epi} Score :{score}'
            )
        )
        score = 0.0
    env.close()
    print(f'Mean of score: {np.mean(scores)}')


if __name__ == '__main__':
    # for i in range(9, 11):
    #     main(0.01 * i)

    epoch = 2000  # options: 2000, 1000, 0

    ckpt_path = 'checkpoints/sac-3-%d.pt' % epoch
    layers = [3, 128, 128]
    # initialize model
    ckpt = torch.load(ckpt_path)
    model = PolicyNet(lr_pi, layers)
    print(layers)
    print('Load weights from %s' % ckpt_path)
    model.load_state_dict(ckpt['policy'])

    for i in range(11):
        eval_sac(model, 0.01 * i)






