import gzip
import numpy as np

from typing import Optional
from .loader import DataLoader


def MINST_loader(conv: Optional[bool] = False, batch_size: Optional[int] = 32) -> tuple[DataLoader, DataLoader]:
    """

    :return: MNIST data used for the fully connected layer and Convolutional layer.
    """
    shape = (28, 28) if conv else (784,)

    path0 = './data/MNIST/raw/'
    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']

    path = [path0 + each for each in files]

    with gzip.open(path[0], 'rb') as path1:
        train_labels = np.frombuffer(path1.read(), np.uint8, offset=8)
    with gzip.open(path[1], 'rb') as path2:
        train_images = np.frombuffer(path2.read(), np.uint8, offset=16).reshape((len(train_labels),) + shape).astype(
            np.float64)
    with gzip.open(path[2], 'rb') as path3:
        test_labels = np.frombuffer(path3.read(), np.uint8, offset=8)
    with gzip.open(path[3], 'rb') as path4:
        test_images = np.frombuffer(path4.read(), np.uint8, offset=16).reshape((len(test_labels),) + shape).astype(
            np.float64)
    if conv:
        train_images = np.expand_dims(train_images, axis=-1)
        test_images = np.expand_dims(test_images, axis=-1)

    def get_one_hot(labels):
        one_hot = np.zeros((len(labels), 10))
        for i in range(len(labels)):
            one_hot[i, labels[i]] = 1
        return one_hot

    train_one_hot = get_one_hot(train_labels).astype(np.float64)
    test_one_hot = get_one_hot(test_labels).astype(np.float64)

    return DataLoader(train_images, train_one_hot, batch_size=batch_size), DataLoader(test_images, test_one_hot,
                                                                                      batch_size=batch_size)
