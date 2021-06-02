import numpy as np
from scipy.linalg import solve_discrete_are


class LQR(object):
    def __init__(self, A, B, Q, R, W, dim=3):
        '''
        :param A: n x n
        :param B: n x m
        :param Q: n x n, PD
        :param R: m x m, PD
        :param dim: n
        '''
        self.A = A
        self.B = B
        self.dim = dim
        self.Q = Q
        self.R = R
        self.W = W
        self.cost = 0.0

    @staticmethod
    def quad(u, A):
        return u.T @ A @ u

    def reset(self):
        self.state = np.random.randn(self.dim) * 5
        # self.state = np.zeros(self.dim)
        return self.state

    def step(self, u):
        noise = np.random.multivariate_normal(mean=np.zeros_like(self.state),
                                              cov=self.W)
        new_state = self.A @ self.state + self.B @ u + noise
        cost = LQR.quad(new_state, self.Q) + LQR.quad(u, self.R)
        self.cost += cost
        self.state = new_state
        return new_state, -cost, False, {}

    def optimum(self):
        P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        K = - np.linalg.inv(self.R + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A
        op_cost = np.trace(P @ self.W)
        La = np.linalg.norm(K, 2)
        return P, K, op_cost, La
