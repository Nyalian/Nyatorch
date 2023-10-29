import numpy as np

from module import Module
from numpy import ndarray
import torch.nn as nn


class LinearLayer(Module):

    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.random.rand(out_features, in_features)
        self.gradient = np.zeros_like(self.weights)

    def forward(self, input: ndarray) -> ndarray:
        return self.weights @ input

    def backward(self, input: ndarray) -> ndarray:
         return input @ self.weights

