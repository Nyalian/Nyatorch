import numpy as np
from module import Module
from numpy import ndarray


class Activation(Module):
    """
    Base class for activation function layers.
    """

    def __init__(self):
        super().__init__()
        self.inputs = None

    def forward(self, inputs: ndarray):
        self.inputs = inputs

    def backward(self, delta: ndarray):
        """
        Backpropagation of the activation layer.

        :param delta: [batch_size, out_feature] or [batch_size, width, height, channel]
        :return: [batch_size, out_feature] or [batch_size, width, height, channel]
        """
        return self._gradient() * delta

    def _gradient(self):
        """
        Gradient of the activation layer. Need to be overridden.
        """
        raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required \"_gradient\" function")


class ReLU(Activation):
    """
    ReLU activation function layer.
    """

    def __init__(self):
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

    def __init__(self):
        super().__init__()

    def forward(self, inputs: ndarray) -> ndarray:
        super().forward(inputs)
        return 1 / (1 + np.exp(-inputs))

    def _gradient(self) -> ndarray:
        return self.forward(self.inputs) * (1 - self.forward(self.inputs))


class Tanh(Activation):
    def __init__(self):
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

    def __init__(self):
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
    def __init__(self):
        super().__init__()

    def forward(self, inputs: ndarray) -> ndarray:
        super().forward(inputs)
        return np.array([1 if x >= 0 else -1 for x in inputs])


class MLPAct(Activation):
    """
    Activation function for MLP.
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs: ndarray) -> ndarray:
        super().forward(inputs)
        return inputs >= 0
