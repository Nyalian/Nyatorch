import numpy as np


class Loss:
    """
    Base class for all loss functions.
    """

    def __init__(self):
        pass

    def calculate(self, prediction, target):
        pass

    def gradient(self, prediction, target):
        pass


class MeanSquaredError(Loss):
    def calculate(self, prediction, target):
        batch_size = prediction.shape[0]
        return 0.5 * np.sum((prediction - target) ** 2) / batch_size

    def gradient(self, prediction, target):
        batch_size = prediction.shape[0]
        return (prediction - target) / batch_size


class CrossEntropy(Loss):

    @staticmethod
    def _softmax(z):
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def calculate(self, prediction, target):
        """
        Calculate CrossEntropyLoss.

        :param prediction: [batch_size, labels]
        :param target: [batch_size, labels]
        :return: The loss value (float)
        """
        batch_size = prediction.shape[0]
        prediction = self._softmax(prediction)
        return -(target * np.log(prediction)).sum() / batch_size

    def gradient(self, prediction, target):
        """
        Calculate CrossEntropyLoss gradient.

        :param prediction: [batch_size, labels]
        :param target: [batch_size, labels]
        :return: The gradient [batch_size, labels]
        """
        prediction = self._softmax(prediction)
        batch_size = prediction.shape[0]
        return (prediction - target) / batch_size
