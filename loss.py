import numpy as np


class Loss:
    def __init__(self):
        pass

    def calculate(self, prediction, target):
        pass

    def gradient(self, prediction, target):
        pass


class MeanSquaredError(Loss):
    def calculate(self, prediction, target):
        return 0.5 * np.sum((prediction - target) ** 2)

    def gradient(self, prediction, target):
        return prediction - target

class CrossEntropy(Loss):

    def _softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)

    def calculate(self, prediction, target):
        m = prediction.shape[1]
        prediction = self._softmax(prediction)
        return -(target * np.log(prediction)).sum() / m

    def gradient(self, prediction, target):
        prediction = self._softmax(prediction)
        m = prediction.shape[1]
        return (prediction - target) / m
