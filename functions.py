import numpy as np


def ReLU(x):
    return np.maximum(x, 0)


def derivative_ReLU(x):
    return np.where(x <= 0, 0, 1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def no_activation(x):
    return x


def derivative_no_activation(x):
    return np.ones_like(x)


def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)


def bin_cross_entropy(y_hat, y):
    return -np.log(y_hat) if y[0] == 1 else -np.log(1 - y_hat)


def derivative_bin_cross_entropy(y_hat, y):
    return -y / y_hat + (1 - y) / (1 - y_hat)


# Data generate function
def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []

    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414

        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)

    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_cube(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []

    for pt in pts:
        inputs.append([pt[0], pt[1]])

        if pt[0] ** 3 > pt[1]:
            labels.append(0)
        else:
            labels.append(1)

    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)

        if 0.1 * i == 0.5:
            continue

        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)


functionDic = {
    'sigmoid': {
        'normal': sigmoid,
        'derivative': derivative_sigmoid,
    },
    'ReLU': {
        'normal': ReLU,
        'derivative': derivative_ReLU,
    },
    'cross_entropy': {
        'normal': bin_cross_entropy,
        'derivative': derivative_bin_cross_entropy,
    },
    'none': {
        'normal': no_activation,
        'derivative': derivative_no_activation,
    },
    'generateData': {
        'linear': generate_linear,
        'XOR': generate_XOR_easy,
        'nonlinear': generate_cube,
    }
}
