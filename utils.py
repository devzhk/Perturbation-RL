import numpy as np
import torch.nn as nn


def random_AB(state_dim=2, action_dim=1, dt=0.1):
    A = np.random.randn(state_dim, state_dim)
    A = A / np.max(np.abs(np.linalg.eigvals(A)))
    B = np.random.randn(state_dim, action_dim)
    return A, B


def get_AB(state_dim=2, action_dim=1, dt=0.1):
    tmp = np.zeros((state_dim, state_dim))
    tmp[: -1, 1:] = np.eye(state_dim - 1) * dt
    A = np.eye(state_dim) + tmp

    B = np.zeros((state_dim, action_dim))
    B[-1, :] = dt
    return A, B


def linear_layer(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU()
    )


def linear_layers(layers):
    nnlist = [linear_layer(in_dim, out_dim) for in_dim, out_dim in zip(layers[:-1], layers[1:])]
    return nn.Sequential(*nnlist)
