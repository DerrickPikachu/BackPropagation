import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)


functionDic = {
    'sigmoid': {
        'normal': sigmoid,
        'derivative': derivative_sigmoid,
    }
}
