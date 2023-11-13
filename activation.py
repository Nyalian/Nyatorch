import numpy as np
from module import Module
from numpy import ndarray


class Activation(Module):
    pass


class ReLU(Activation):

    def __init__(self):
        pass

    def forward(self, input: ndarray) -> ndarray:
        return input * (input > 0)

    def backward(self, input: ndarray) -> ndarray:
        return (input > 0).astype(float)


class Sigmoid(Activation):

    def __init__(self):
        pass

    def forward(self, input: ndarray) -> ndarray:
        return 1 / (1 + np.exp(-input))

    def backward(self, input: ndarray) -> ndarray:
        return self.forward(input) * (1 - self.forward(input))


class Linear(Activation):

    def __init__(self):
        pass

    def forward(self, input: ndarray) -> ndarray:
        return input

    def backward(self, input: ndarray) -> ndarray:
        return np.ones_like(input).astype(float)


class Hebb_act(Activation):

    def __init__(self):
        pass

    def forward(self, input: ndarray) -> ndarray:
        return np.array([1 if x >= 0 else -1 for x in input])


class MLP_act(Activation):

    def __init__(self):
        pass

    def forward(self, input: ndarray) -> ndarray:
        return input >= 0
