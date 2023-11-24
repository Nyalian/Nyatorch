import numpy as np
from numpy import ndarray


class Loss:
    """
    Base class for all loss functions.
    """

    def __init__(self):
        pass

    def calculate(self, prediction, label):
        pass

    def gradient(self, prediction, label):
        pass


class MeanSquaredError(Loss):
    def calculate(self, prediction, label) -> float:
        batch_size = prediction.shape[0]
        return 0.5 * np.sum((prediction - label) ** 2) / batch_size

    def gradient(self, prediction, label) -> ndarray:
        batch_size = prediction.shape[0]
        return (prediction - label) / batch_size


class CrossEntropy(Loss):

    @staticmethod
    def _softmax(z) -> ndarray:
        z -= np.max(z, axis=1, keepdims=True)
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def calculate(self, prediction, label) -> float:
        """
        Calculate CrossEntropyLoss.

        :param prediction: [batch_size, labels]
        :param label: [batch_size, labels]
        :return: The loss value (float)
        """
        batch_size = prediction.shape[0]
        prediction = self._softmax(prediction)
        return -(label * np.log(prediction + 1e-6)).sum() / batch_size

    def gradient(self, prediction, label) -> ndarray:
        """
        Calculate CrossEntropyLoss gradient.

        :param prediction: [batch_size, labels]
        :param label: [batch_size, labels]
        :return: The gradient [batch_size, labels]
        """
        prediction = self._softmax(prediction)
        batch_size = prediction.shape[0]
        return (prediction - label) / batch_size
