import numpy as np
from numpy import ndarray

from .abstract import Module


class LinearLayer(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        """
        Initialize the fully connected layer.
        The size of the weight is [in_features, out_features].
        The size of the bias is [1, out_features].

        :param in_features: The input features
        :param out_features: The output features
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        bound = np.sqrt(6. / (self.in_features + self.out_features))
        self.weights = np.random.uniform(-bound, bound, (self.in_features, self.out_features))
        self.bias = np.random.rand(1, out_features)
        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias = np.zeros_like(self.bias)
        self.inputs = None

    def forward(self, inputs: ndarray) -> ndarray:
        """
        Forward calculation of the fully connected layer.

        :param inputs: [batch_size, in_features]
        :return: [batch_size, out_features]
        """
        self.inputs = inputs
        return inputs @ self.weights + self.bias

    def backward(self, delta: ndarray) -> ndarray:
        """
        Backpropagation of the fully connected layer.

        :param delta: [batch_size, out_features]
        :return: [batch_size, in_features]
        """
        self.gradient_weights = self.inputs.T @ delta
        self.gradient_bias = delta.sum(axis=0, keepdims=True)
        return delta @ self.weights.T

    def get_parameter(self) -> dict[str, ndarray]:
        """
        Get the parameter of the fully connected layer.
        :return: the parameter
        """
        return {"weights": self.weights, "bias": self.bias}

    def set_parameter(self, parameter: dict[str, ndarray]) -> None:
        """
        Set the parameter of the fully connected layer.

        :param parameter:
        """
        self.weights = parameter["weights"]
        self.bias = parameter["bias"]

    def update(self, learning_rate: float) -> None:
        """
        Update the weights and bias.

        :param learning_rate: the learning_rate
        """
        self.weights -= learning_rate * self.gradient_weights
        self.bias -= learning_rate * self.gradient_bias


class Flatten(Module):

    def __init__(self) -> None:
        super().__init__()
        self.shape = None

    def forward(self, inputs: ndarray) -> ndarray:
        self.shape = inputs.shape
        batch_size, height, width, channel = inputs.shape
        return (inputs.flatten()).reshape(batch_size, height * width * channel)

    def backward(self, delta: ndarray) -> ndarray:
        return delta.reshape(self.shape)
