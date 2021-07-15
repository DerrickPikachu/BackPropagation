import numpy as np
from layer import *
from functions import *


class Network:
    def __init__(self):
        self.inputLayer = None
        self.hiddenLayer = []
        self.outputLayer = None

    def addInputLayer(self, dimension) -> None:
        self.inputLayer = InputLayer(size=dimension)

    def addHiddenLayer(self, neurons, activation) -> None:
        pass

    def addOutputLayer(self, dimension, activation) -> None:
        pass

    def forwardPass(self):
        self.inputLayer = InputLayer(size=2)
        hiddenLayer = HiddenLayer(size=2, activation=sigmoid)
        self.outputLayer = OutputLayer(size=2, activation=sigmoid)

        self.inputLayer.setup([1, 2])
        weights1 = np.array([[0.5, 0.5], [0.5, 0.5]])
        weights2 = np.array([[2, 3], [4, 5]])

        hiddenLayer.forward(inputs=self.inputLayer.outputValue, weight=weights1)
        self.outputLayer.forward(hiddenLayer.outputValue, weights2)
        print(self.outputLayer.outputValue)

    def fit(self, inputs, labels, loss, batch_size=20, epoch=100) -> []:
        pass


if __name__ == "__main__":
    myNet = Network()
    myNet.forwardPass()
