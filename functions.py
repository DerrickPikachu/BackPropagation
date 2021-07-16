import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)


def cross_entropy(y_hat, y):
    return y @ (-np.log(y_hat))


def derivative_cross_entropy(y_hat, y):
    return -(y / y_hat)


functionDic = {
    'sigmoid': {
        'normal': sigmoid,
        'derivative': derivative_sigmoid,
    },
    'cross_entropy': {
        'normal': cross_entropy,
        'derivative': derivative_cross_entropy,
    }
}
