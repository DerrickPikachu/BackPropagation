from network import Network
from functions import *
import numpy as np
import pickle
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

    def saveModel(self, filename):
        pickle.dump(self.model, open(filename, 'wb'))

    def loadModel(self, filename):
        self.model = pickle.load(open(filename, 'rb'))

    def show(self):
        # Choose the training data
        print("Chose the data set:")
        print("1. linear")
        print("2. XOR")
        print("3. nonlinear")
        choice = int(input())

        # Generate data
        if choice == 1:
            x, labels = functionDic['generateData']['linear'](n=500)
        elif choice == 2:
            x, labels = functionDic['generateData']['XOR']()
        else:
            x, labels = functionDic['generateData']['nonlinear'](n=1000)

        # Training or testing?
        print("DO you want to use saved network?")
        print("1. use saved network, 2. train a new one")
        choice = int(input())

        if choice == 1:
            # Load the network
            print('Which file you want to read?')
            filename = input()
            self.loadModel(filename)

            pred_y = self.model.predict(inputs=x)

            hits = 0
            for i in range(len(labels)):
                if pred_y[i] == labels[i]:
                    hits = hits + 1
            print("hit rate: {}".format(hits / len(labels)))

            self.show_result(x, labels, pred_y)
        else:
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

            # Saving the network
            print('Do you want to save the network?')
            print('1. yes, 2. no')
            choice = int(input())
            if choice == 1:
                print('What is the file name?')
                filename = input()
                self.saveModel(filename)
