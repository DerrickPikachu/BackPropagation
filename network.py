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
        if self.inputLayer is None:
            # Throw exception when the input layer haven't been specified
            raise RuntimeError('Add hidden layer without input layer') from exec
        elif len(self.hiddenLayer) == 0:
            self.hiddenLayer.append(HiddenLayer(
                pre_size=self.inputLayer.size(),
                size=neurons,
                activation=functionDic[activation]
            ))
        else:
            self.hiddenLayer.append(HiddenLayer(
                pre_size=self.hiddenLayer[-1].size(),
                size=neurons,
                activation=functionDic[activation]
            ))

    def addOutputLayer(self, neurons, activation) -> None:
        if len(self.hiddenLayer) == 0:
            if self.inputLayer is None:
                # Throw exception when the input layer haven't been specified
                raise RuntimeError('Add output layer without hidden layer or input layer') from exec
            else:
                self.outputLayer = OutputLayer(
                    pre_size=self.inputLayer.size(),
                    size=neurons,
                    activation=functionDic[activation]
                )
        else:
            self.outputLayer = OutputLayer(
                pre_size=self.hiddenLayer[-1].size(),
                size=neurons,
                activation=functionDic[activation]
            )

    def forwardPass(self):
        pass
        # self.inputLayer = InputLayer(size=2)
        # hiddenLayer = HiddenLayer(size=2, activation=sigmoid)
        # self.outputLayer = OutputLayer(size=2, activation=sigmoid)
        #
        # self.inputLayer.setup([1, 2])
        # weights1 = np.array([[0.5, 0.5], [0.5, 0.5]])
        # weights2 = np.array([[2, 3], [4, 5]])
        #
        # hiddenLayer.forward(inputs=self.inputLayer.outputValue, weight=weights1)
        # self.outputLayer.forward(hiddenLayer.outputValue, weights2)
        # print(self.outputLayer.outputValue)

    def fit(self, inputs, labels, loss, batch_size=20, epoch=100) -> []:
        pass


if __name__ == "__main__":
    myNet = Network()
    myNet.forwardPass()
