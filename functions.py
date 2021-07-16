import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)


def bin_cross_entropy(y_hat, y):
    return -np.log(y_hat) if y[0] == 1 else -np.log(1 - y_hat)


def derivative_bin_cross_entropy(y_hat, y):
    return -y / y_hat + (1 - y) / (1 - y_hat)


functionDic = {
    'sigmoid': {
        'normal': sigmoid,
        'derivative': derivative_sigmoid,
    },
    'cross_entropy': {
        'normal': bin_cross_entropy,
        'derivative': derivative_bin_cross_entropy,
    }
}
