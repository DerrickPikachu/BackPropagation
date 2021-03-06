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

        firstGradient = functionDic[loss_f]['derivative'](y_hat=outputDistribution, y=labels)

        if len(self.hiddenLayer) == 0:
            gradient = self.outputLayer.backward(self.inputLayer.outputValue, firstGradient)
        else:
            gradient = self.outputLayer.backward(self.hiddenLayer[-1].outputValue, firstGradient)

        for i in range(len(self.hiddenLayer) - 1, -1, -1):
            if i == 0:
                gradient = self.hiddenLayer[i].backward(self.inputLayer.outputValue, gradient)
            else:
                gradient = self.hiddenLayer[i].backward(self.hiddenLayer[i-1].outputValue, gradient)

    def update(self):
        for hidden in self.hiddenLayer:
            hidden.update()
        self.outputLayer.update()

    def predict(self, inputs):
        pred_y = []
        pred_prob_y = []
        for i in inputs:
            self.inputLayer.setup(i)
            self.forwardPass()
            pred_prob_y.append(self.outputLayer.outputValue)
            pred_y.append(1 if self.outputLayer.outputValue[0] > 0.5 else 0)
        return np.array(pred_y), np.array(pred_prob_y)

    def fit(self, inputs, labels, loss_f, batch_size=15, epoch=100) -> []:
        lossRecord = []
        for i in range(1, epoch + 1):
            lossSum = 0
            hits = 0
            trainOrder = np.arange(len(inputs))
            np.random.shuffle(trainOrder)

            counter = 0
            for index in trainOrder:
                self.inputLayer.setup(inputs[index])
                self.forwardPass()

                if (labels[index] == 1 and self.outputLayer.outputValue > 0.5)\
                        or (labels[index] == 0 and self.outputLayer.outputValue <= 0.5):
                    hits = hits + 1

                loss = functionDic[loss_f]['normal'](
                    y_hat=self.outputLayer.outputValue,
                    y=labels[index]
                )
                lossSum = lossSum + loss

                self.backwardPass(loss_f, labels[index])

                counter = counter + 1
                if counter == batch_size:
                    self.update()
                    counter = 0

            self.update()
            if i % 100 == 0:
                print("epoch {} loss: {} hit rate: {}".format(i, lossSum / len(inputs), hits / len(inputs)))

            lossRecord.append(lossSum / len(inputs))

        return epoch, np.array(lossRecord)


if __name__ == "__main__":
    myNet = Network()
    myNet.addInputLayer(dimension=2)
    myNet.addHiddenLayer(neurons=2, activation="sigmoid")
    myNet.addOutputLayer(neurons=1, activation="sigmoid")
    myNet.inputLayer.setup(np.array([1, 0]))
    myNet.forwardPass()
    myNet.backwardPass('cross_entropy', np.array([1]))
    print(myNet.outputLayer.outputValue)
