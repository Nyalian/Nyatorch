import numpy as np

from module import Module
from numpy import ndarray
import torch.nn as nn


class Linear(Module):

    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.zeros((out_features, in_features))

    def forward(self, input: ndarray) -> ndarray:
        return self.weight @ input.T
