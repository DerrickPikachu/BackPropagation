import numpy as np


class Layer:
    def __init__(self, size):
        self.weightedSum = None
        # The last one is used to multiply the bias weight
        self.outputValue = np.zeros(size + 1)
        self.outputValue[-1] = 1

    def forward(self, inputs):
        pass

    def backward(self, inputs, back_gradient):
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
        self.weights = np.random.uniform(-2, 2, (size, pre_size))
        # self.weights = np.array([[0.5, 0.5, 0], [0.5, 0.5, 0]])
        self.gradientMatrix = np.zeros((size, pre_size))

        for i in range(size):
            self.weights[i][-1] = np.random.uniform(0, 0.3)

    def forward(self, inputs):
        self.weightedSum = self.weights @ inputs
        self.outputValue = np.append(self.activation['normal'](self.weightedSum), np.array([1]))

    def backward(self, inputs, back_gradient):
        if len(self.outputValue) == 1:
            activationGradient = self.activation['derivative'](self.outputValue)
        else:
            activationGradient = self.activation['derivative'](self.outputValue[:-1])
        weightedSumGradient = activationGradient * back_gradient

        graMatrix = None
        for element in inputs:
            col = element * weightedSumGradient
            if graMatrix is None:
                graMatrix = col
            else:
                graMatrix = np.c_[graMatrix, col]

        self.gradientMatrix = self.gradientMatrix + graMatrix

        return weightedSumGradient @ self.weights[:, :-1]

    def update(self, alpha=0.1):
        self.weights = self.weights - alpha * self.gradientMatrix
        self.gradientMatrix = np.zeros(self.weights.shape)


class OutputLayer(HiddenLayer):
    def __init__(self, pre_size, size, activation):
        super().__init__(pre_size, size, activation)
        # self.weights = np.array([[0.5, 0.5, 0]])

    def forward(self, inputs):
        self.weightedSum = self.weights @ inputs
        if type(self.weightedSum) != type(inputs):
            self.weightedSum = np.array([self.weightedSum])
        self.outputValue = self.activation['normal'](self.weightedSum)

