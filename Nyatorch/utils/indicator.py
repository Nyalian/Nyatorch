import numpy as np
from numpy import ndarray

from Nyatorch.utils import confusion_matrix


def confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    num_classes = len(classes)
    cm = np.zeros((num_classes, num_classes))

    for i in range(len(y_true)):
        true_class = np.where(classes == y_true[i])[0][0]
        pred_class = np.where(classes == y_pred[i])[0][0]
        cm[true_class, pred_class] += 1

    return cm


def accuracy(prediction: ndarray, label: ndarray) -> float:
    """
    Compute the accuracy of predictions and labels, the input should be two sets of one-hot encoded data.

    :param prediction: the prediction value [features, samples]
    :param label: the label value [features, samples]
    :return: the accuracy
    """
    predict_idx = np.argmax(prediction, axis=1)
    label_idx = np.argmax(label, axis=1)
    return np.mean(predict_idx == label_idx) * 100


def precision(prediction: ndarray, label: ndarray) -> float:
    """
    Compute the precision of predictions and labels, the input should be two sets of one-hot encoded data.

    :param prediction: the prediction value [features, samples]
    :param label: the label value [features, samples]
    :return: the precision
    """
    predict_idx = np.argmax(prediction, axis=1)
    label_idx = np.argmax(label, axis=1)

    cm = confusion_matrix(label_idx, predict_idx)
    true_positive = cm[1, 1]
    false_positive = cm[0, 1]
    prec = true_positive / (true_positive + false_positive)

    return prec * 100



def f_score(prediction: ndarray, label: ndarray) -> float:
    """
    Compute the F1 score of predictions and labels, the input should be two sets of one-hot encoded data.

    :param prediction: the prediction value [features, samples]
    :param label: the label value [features, samples]
    :return: the F1 score
    """
    predict_idx = np.argmax(prediction, axis=1)
    label_idx = np.argmax(label, axis=1)

    cm = confusion_matrix(label_idx, predict_idx)
    true_positive = cm[1, 1]
    false_positive = cm[0, 1]
    false_negative = cm[1, 0]

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    f_score = 2 * (precision * recall) / (precision + recall)

    return f_score
