import numpy as np
from module import Module
from numpy import ndarray

class ReLU(Module):

    def __init__(self):
        pass

    def forward(self, input: ndarray) -> ndarray:

        return np.maximum(0, input)

class Sigmoid(Module):

    def __init__(self):
        pass

    def forward(self, input: ndarray) -> ndarray:
        return 1 / (1 + np.exp(-input))