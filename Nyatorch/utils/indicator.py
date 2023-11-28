import numpy as np
from numpy import ndarray
from sklearn.metrics import confusion_matrix as conf_matrix


def confusion_matrix(test_y: ndarray, pred: ndarray) -> ndarray:
    cm = conf_matrix(y_true=np.argmax(test_y, axis=1), y_pred=np.argmax(pred, axis=1))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100.0

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


def precision(label: ndarray, prediction: ndarray, cla_num: int) -> float:
    """
    Calculate precision for a specific class.

    :param label: True labels in one-hot encoded format.
    :param prediction: Predicted labels in one-hot encoded format.
    :param cla_num: Index of the class for which precision is calculated.
    :return: Precision for the specified class.
    """
    predict_idx = np.argmax(prediction, axis=1)
    label_idx = np.argmax(label, axis=1)

    cm = conf_matrix(label_idx, predict_idx)
    print(cm)
    true_positive = cm[cla_num][cla_num]

    # Sum of true positives for other classes
    false_negative = np.sum(cm[cla_num]) - true_positive

    # Sum of true negatives for other classes
    false_positive = np.sum(cm[:, cla_num]) - true_positive

    # Sum of true negatives for all other classes
    true_negative = np.sum(cm, axis=(0, 1)) - true_positive - false_negative - false_positive

    precision_value = true_positive / (true_positive + false_positive)
    return precision_value


def recall(label: ndarray, prediction: ndarray, cla_num: int) -> float:
    """
    Calculate recall for a specific class.

    :param label: True labels in one-hot encoded format.
    :param prediction: Predicted labels in one-hot encoded format.
    :param cla_num: Index of the class for which recall is calculated.
    :return: Recall for the specified class.
    """
    predict_idx = np.argmax(prediction, axis=1)
    label_idx = np.argmax(label, axis=1)

    cm = conf_matrix(label_idx, predict_idx)
    print(cm)
    true_positive = cm[cla_num][cla_num]

    # Sum of true positives for other classes
    false_negative = np.sum(cm[cla_num]) - true_positive

    # Sum of true negatives for other classes
    false_positive = np.sum(cm[:, cla_num]) - true_positive

    # Sum of true negatives for all other classes
    true_negative = np.sum(cm, axis=(0, 1)) - true_positive - false_negative - false_positive

    recall_value = true_positive / (true_positive + false_positive)
    return recall_value


def f_score(label: ndarray, prediction: ndarray, cla_num: int) -> float:
    """
    Calculate F1 score for a specific class.

    :param label: True labels in one-hot encoded format.
    :param prediction: Predicted labels in one-hot encoded format.
    :param cla_num: Index of the class for which F1 score is calculated.
    :return: F1 score for the specified class.
    """
    precision_value = precision(label, prediction, cla_num)
    recall_value = recall(label, prediction, cla_num)

    return 2 * precision_value * recall_value / (precision_value + recall_value)