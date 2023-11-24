import numpy as np
from numpy import ndarray

from .abstract import Activation


class ReLU(Activation):
    """
    ReLU activation function layer.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: ndarray) -> ndarray:
        super().forward(inputs)
        return inputs * (inputs > 0)

    def _gradient(self) -> ndarray:
        return (self.inputs > 0).astype(float)


class Sigmoid(Activation):
    """
    Sigmoid activation function layer.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: ndarray) -> ndarray:
        super().forward(inputs)
        return 1 / (1 + np.exp(-inputs))

    def _gradient(self) -> ndarray:
        return self.forward(self.inputs) * (1 - self.forward(self.inputs))


class Tanh(Activation):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: ndarray) -> ndarray:
        super().forward(inputs)
        return (np.exp(inputs) - np.exp(-inputs)) / (np.exp(inputs) + np.exp(-inputs))

    def _gradient(self) -> ndarray:
        return 1 - self.forward(self.inputs) * self.forward(self.inputs)


class Linear(Activation):
    """
    Linear activation function layer.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: ndarray) -> ndarray:
        super().forward(inputs)
        return inputs

    def _gradient(self) -> ndarray:
        return np.ones_like(self.inputs).astype(float)


class HebbAct(Activation):
    """
    Activation function for Hebb Learning algorithm.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: ndarray) -> ndarray:
        super().forward(inputs)
        return np.array([1 if x >= 0 else -1 for x in inputs[0]]).reshape(-1,1)

    def _gradient(self):
        pass


class MLPAct(Activation):
    """
    Activation function for MLP.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: ndarray) -> ndarray:
        super().forward(inputs)
        return inputs >= 0

    def _gradient(self):
        pass