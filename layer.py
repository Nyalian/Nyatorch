import numpy as np

from module import Module
from numpy import ndarray
import torch.nn as nn

class Layer:
    pass

class LinearLayer(Layer):

    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.random.rand(out_features, in_features)
        self.gradient = np.zero_like(self.weights)

    def forward(self, input: ndarray) -> ndarray:
        return self.weights @ input.T

    def backward(self, input: ndarray) -> ndarray:
         return input @ self.weights

