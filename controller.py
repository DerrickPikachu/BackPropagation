from network import Network
from functions import *
import numpy as np
import matplotlib.pyplot as plt


class Interface:
    def __init__(self):
        self.model = Network()

    def buildNetwork(self):
        self.model = Network()
        self.model.addInputLayer(dimension=2)
        self.model.addHiddenLayer(neurons=2, activation="sigmoid")
        self.model.addHiddenLayer(neurons=2, activation="sigmoid")
        # self.model.addHiddenLayer(neurons=2, activation="ReLU")
        # self.model.addHiddenLayer(neurons=2, activation="ReLU")
        self.model.addOutputLayer(neurons=1, activation="sigmoid")

    def show_result(self, x, y, pred_y):
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

    def show_learning_curve(self, epochs, loss):
        plt.subplot(1, 1, 1)
        plt.title('Learning curve', fontsize=18)
        plt.plot(epochs, loss)
        plt.show()

    def show(self):
        # Training or testing?
        print("DO you want to use saved network?")
        print("1. use saved network, 2. train a new one")
        choice = int(input())

        if choice == 1:
            pass
        else:
            # Choose the training data
            print("Chose the data set:")
            print("1. linear")
            print("2. XOR")
            choice = int(input())

            # Generate data
            if choice == 1:
                x, labels = functionDic['generateData']['linear']()
            elif choice == 2:
                x, labels = functionDic['generateData']['XOR']()

            # Training
            self.buildNetwork()
            epoch, lossRecord = self.model.fit(
                inputs=x,
                labels=labels,
                loss_f="cross_entropy",
                epoch=3000,
                batch_size=10
            )

            # Show graph
            pred_y = self.model.predict(inputs=x)
            self.show_result(x, labels, pred_y)
            self.show_learning_curve(range(1, epoch + 1), lossRecord)