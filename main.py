from network import Network
import numpy as np
import matplotlib.pyplot as plt


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


def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.show()


# Choose the training data
print("Chose the data set:")
print("1. linear")
print("2. XOR")
choice = int(input())

# Build training data
x = np.array([], float)
labels = np.array([], int)
if choice == 1:
    x, labels = generate_linear()
elif choice == 2:
    x, labels = generate_XOR_easy()

# Build the neuron network
model = Network()
model.addInputLayer(dimension=2)
model.addHiddenLayer(neurons=2, activation="sigmoid")
model.addHiddenLayer(neurons=2, activation="sigmoid")
# model.addHiddenLayer(neurons=2, activation="ReLU")
# model.addHiddenLayer(neurons=2, activation="ReLU")
model.addOutputLayer(neurons=1, activation="sigmoid")

# Training
# lossValue, hitRate = model.fit(inputs=x, labels=labels, loss_f="cross_entropy")
model.fit(inputs=x, labels=labels, loss_f="cross_entropy", epoch=3000, batch_size=10)

# Show graph below
pred_y = model.predict(inputs=x)
show_result(x, labels, pred_y)
