import numpy as np


class Layer:
    def __init__(self, size):
        self.weightedSum = None
        # The last one is used to multiply the bias weight
        self.outputValue = np.zeros(size + 1)
        self.outputValue[-1] = 1

    def forward(self, inputs):
        pass

    def size(self):
        return self.outputValue.size


class InputLayer(Layer):
    def __init__(self, size):
        super().__init__(size)

    # Set the input vector to this layer
    def setup(self, inputs):
        self.outputValue = np.append(inputs, np.array([1]))


class HiddenLayer(Layer):
    def __init__(self, pre_size, size, activation):
        super().__init__(size)
        self.activation = activation
        self.weights = np.random.rand(size, pre_size)

        for i in range(size):
            self.weights[i][-1] = np.random.uniform(0, 0.3)

    def forward(self, inputs):
        self.weightedSum = self.weights @ inputs
        self.outputValue = np.append(self.activation(self.weightedSum), np.array([1]))


class OutputLayer(Layer):
    def __init__(self, pre_size, size, activation):
        super().__init__(size)
        self.activation = activation
        self.weights = np.random.uniform(0., 1., (size, pre_size))

        for i in range(size):
            self.weights[i][-1] = np.random.uniform(0, 0.3)

    def forward(self, inputs):
        self.weightedSum = self.weights @ inputs
        self.outputValue = self.activation(self.weightedSum)
