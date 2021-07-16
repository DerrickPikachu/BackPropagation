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
        tem = self.inputLayer.outputValue
        for hidden in self.hiddenLayer:
            hidden.forward(tem)
            tem = hidden.outputValue
        self.outputLayer.forward(tem)

    def backwardPass(self, loss_f, labels):
        outputDistribution = self.outputLayer.outputValue
        if self.outputLayer.size() == 1:
            outputDistribution = np.array([self.outputLayer.outputValue[0], 1 - self.outputLayer.outputValue[0]])
            labels = np.array([labels[0], 1 - labels[0]])

        firstGradient = functionDic[loss_f]['derivative'](y_hat=outputDistribution, y=labels)

        gradient = None
        if len(self.hiddenLayer) == 0:
            gradient = self.outputLayer.backward(self.inputLayer.outputValue, firstGradient)
        else:
            gradient = self.outputLayer.backward(self.hiddenLayer[-1].outputValue, firstGradient)

        for i in range(len(self.hiddenLayer) - 1, -1, -1):
            if i == 0:
                gradient = self.hiddenLayer[i].backward(self.inputLayer.outputValue, gradient)
            else:
                gradient = self.hiddenLayer[i].backward(self.hiddenLayer[i-1].outputValue, gradient)

    def fit(self, inputs, labels, loss, batch_size=20, epoch=100) -> []:
        batchData = []
        batchDataLen = inputs.shape[0] / batch_size

        # Fill data in each batch
        for i in range(batchDataLen):
            batchData.append(inputs[i * batch_size : (i + 1) * batch_size])

        for i in range(epoch):
            for batch in batchData:
                for data in batch:
                    self.inputLayer.setup(data)
                    self.forwardPass()
                    self.backwardPass()


if __name__ == "__main__":
    myNet = Network()
    myNet.addInputLayer(2)
    myNet.addHiddenLayer(2, "sigmoid")
    myNet.addHiddenLayer(2, "sigmoid")
    myNet.addOutputLayer(1, "sigmoid")

    myNet.inputLayer.setup(np.array([1, 2]))
    myNet.forwardPass()
    myNet.backwardPass('cross_entropy', np.array([0]))
    print("finish")
