import numpy as np
from module import Module
from numpy import ndarray

class ReLU(Module):

    def __init__(self):
        pass

    def forward(self, input: ndarray) -> ndarray:

        return input*(input>0)

    def gradient(self, input: ndarray) -> ndarray:

        return (input>0).astype(float)

class Sigmoid(Module):

    def __init__(self):
        pass

    def forward(self, input: ndarray) -> ndarray:
        return 1 / (1 + np.exp(-input))

    def gradient(self, input: ndarray) -> ndarray:
        return self.forward(input) * (1-self.forward(input))

class Linear(Module):

    def __init__(self):
        pass

    def forward(self, input: ndarray) -> ndarray:

        return input

    def gradient(self, input: ndarray) -> ndarray:

        return np.ones_like(input).astype(float)