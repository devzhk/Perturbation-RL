import math
import numpy as np


import torch
from envs.LQR import LQR
from utils import get_AB


torch.manual_seed(2021)
np.random.seed(2021)

learning_rate = 0.0003
gamma = 0.9
lmbda = 0.9
eps_clip = 0.2
K_epoch = 10
rollout_len = 3
buffer_size = 30
minibatch_size = 32


def PDcontrol(x, K):
    u = K @ x
    return u


def main():
    print_interval = 20
    # create environment
    state_dim = 1
    action_dim = 1
    # A = np.array([[1.0]])

    dt = 0.1
    A, B = get_AB(state_dim, action_dim, dt)

    sigma = 0.1
    W = sigma * np.eye(state_dim)
    # B = np.eye(2)
    Q = np.eye(state_dim) * 10.0
    R = np.eye(action_dim)
    env = LQR(A, B, Q, R, W, state_dim)
    P, K, op_cost, La = env.optimum()
    print('Optimal cost:{}; La: {}'.format(op_cost, La))
    print(f'P: {P};\n K : {K}')
    # print(A + B @ K)
    # print(np.linalg.eigvals(A + B @ K))
    sample_num = 1000
    avg_score = 0.0
    for n_epi in range(sample_num):
        s = env.reset(factor=2.0)
        score = 0.0
        for i in range(1000):
            for t in range(rollout_len):
                a = PDcontrol(s, K)
                s_prime, r, done, info = env.step(a)
                s = s_prime
                score += r
        score /= 1000 * rollout_len
        avg_score += score
        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".
                  format(n_epi, avg_score / print_interval))
            avg_score = 0.0


if __name__ == '__main__':
    main()



