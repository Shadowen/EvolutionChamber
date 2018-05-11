import numpy as np


def sigmoid(x, ):
    return 1 / (1 + np.exp(-x))


def softmax(x, *, temperature=1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum()


def cat_ones(a):
    return np.concatenate([a, np.ones([1, 1])], axis=1)
