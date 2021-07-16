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
        self.weights = np.random.rand(size, pre_size)
        self.gradientMatrix = None

        for i in range(size):
            self.weights[i][-1] = np.random.uniform(0, 0.3)

    def forward(self, inputs):
        self.weightedSum = self.weights @ inputs
        self.outputValue = np.append(self.activation['normal'](self.weightedSum), np.array([1]))

    # TODO: Haven't been test
    def backward(self, inputs, back_gradient):
        activationGradient = self.activation['derivative'](self.weightedSum)
        weightedSumGradient = activationGradient * back_gradient

        self.gradientMatrix = None
        for element in inputs:
            col = element * weightedSumGradient
            if self.gradientMatrix is None:
                self.gradientMatrix = col
            else:
                self.gradientMatrix = np.c_[self.gradientMatrix, weightedSumGradient]

        return weightedSumGradient @ self.weights[:, :-1]

    def update(self, alpha=0.1):
        self.weights = self.weights - alpha * self.gradientMatrix


class OutputLayer(HiddenLayer):
    def __init__(self, pre_size, size, activation):
        super().__init__(pre_size, size, activation)

    def forward(self, inputs):
        self.weightedSum = self.weights @ inputs
        self.outputValue = self.activation['normal'](self.weightedSum)

    def backward(self, inputs, back_gradient):
        gradient = None
        if self.size() == 1:
            gradient = back_gradient[0] if back_gradient[0] != 0 else -back_gradient[1]
        return super().backward(inputs, np.array([gradient]))

