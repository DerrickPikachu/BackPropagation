import numpy as np


class Layer:
    def __init__(self, size):
        self.weightedSum = np.zeros((size, 1))
        self.outputValue = np.zeros((size, 1))

    def forward(self, inputs, weight):
        pass


class InputLayer(Layer):
    def __init__(self, size):
        super().__init__(size)

    # Set the input vector to this layer
    def setup(self, inputs):
        self.outputValue = inputs


class HiddenLayer(Layer):
    def __init__(self, size, activation):
        super().__init__(size)
        self.activation = activation

    def forward(self, inputs, weight):
        self.weightedSum = weight @ inputs
        self.outputValue = self.activation(self.weightedSum)


class OutputLayer(Layer):
    def __init__(self, size, activation):
        super().__init__(size)
        self.activation = activation

    def forward(self, inputs, weight):
        self.weightedSum = weight @ inputs
        self.outputValue = self.activation(self.weightedSum)
