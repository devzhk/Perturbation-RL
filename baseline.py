from envs.LQR import LQR
import numpy as np


if __name__ == '__main__':

    state_dim = 2
    action_dim = 1

    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    sigma = 0.1
    W = sigma * np.eye(state_dim)
    # B = np.eye(2)
    Q = np.eye(state_dim)
    R = np.eye(action_dim) * 10.0
    env = LQR(A, B, Q, R, W, state_dim)

    P, K, op_cost, La = env.optimum()
    print('Optimal cost:{}; La: {}'.format(op_cost, La))
    print(f'P: {P};\n K : {K}')
