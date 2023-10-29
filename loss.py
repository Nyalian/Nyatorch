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
