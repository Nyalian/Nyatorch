from abc import ABCMeta, abstractmethod
from typing import Any

from numpy import ndarray


class Module(metaclass=ABCMeta):

    def __init__(self):
        pass

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs)

    @abstractmethod
    def forward(self, *inputs: Any):
        raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required \"forward\" function")

    @abstractmethod
    def backward(self, *delta: Any):
        raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required \"backward\" function")

    def get_parameter(self):
        return {}

    def set_parameter(self, parameter: dict):
        pass

    def update(self, learning_rate):
        pass


class ConvNd(Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int, padding: int, stride: int) -> None:
        """
        Initialize the ConvNd layer.

        :param in_channel: the in_channel
        :param out_channel: the out_channel
        :param kernel_size: the kernel_size
        :param padding: the padding
        :param stride: the stride
        """
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.weights = None
        self.bias = None
        self.gradient_weights = None
        self.gradient_bias = None

    @abstractmethod
    def forward(self, inputs: ndarray):
        pass

    @abstractmethod
    def backward(self, delta: ndarray):
        pass


class Activation(Module, metaclass=ABCMeta):
    def __init__(self) -> None:
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

    @abstractmethod
    def _gradient(self):
        """
        Gradient of the activation layer. Need to be overridden.
        """
        pass
