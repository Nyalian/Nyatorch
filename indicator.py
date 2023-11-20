import numpy as np
from numpy import ndarray


def accuracy(predict: ndarray, label: ndarray) -> float:
    """
    Compute the accuracy of predictions and labels, the input should be two sets of one-hot encoded data.

    :param predict: the predict value [features, samples]
    :param label: the label value [features, samples]
    :return: the accuracy
    """
    predict_idx = np.argmax(predict, axis=1)
    label_idx = np.argmax(label, axis=1)
    return np.mean(predict_idx == label_idx) * 100
