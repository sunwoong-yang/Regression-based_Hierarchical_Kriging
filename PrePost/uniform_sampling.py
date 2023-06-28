import numpy as np


def uniform(in_dim, n_pts):
    samples = np.random.uniform(size=(n_pts, in_dim))
    return samples