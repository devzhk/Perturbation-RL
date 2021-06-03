import math
import numpy as np

import gym
import torch
import torch.optim as optim
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


def transform(theta, thetadot):
    return torch.cat([torch.cos(theta), torch.sin(theta), thetadot])


def estimate(model, func, num_trial=5, iter_num=50000):
    La_list = []
    for i in range(num_trial):
        la = func(model, iter_num)
        print('Trial :{}, La: {}'.format(i, la))
        La_list.append(la)
    Las = np.array(La_list)
    mean = Las.mean()
    std = Las.std()
    return mean, std


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


def main():
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
            state = env.state
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


if __name__ == '__main__':
    main()



