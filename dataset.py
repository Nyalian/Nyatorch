import numpy as np
from numpy import ndarray
import gzip


class Dataset:
    def __init__(self, sample: ndarray, label: ndarray, batch_size=1, shuffle=True):
        """
        Initialize the incoming data and perform other operations.

        :param sample: The sample data. It should be [samples, features] or [samples, width, height, channel].
        :param label: The label data. It should be [samples, features]
        :param batch_size: The number of samples used for each model update.
        :param shuffle: Shuffle the order of samples and labels.
        """
        assert sample.shape[0] == label.shape[0], \
            f"The number of samples and labels does not match: sample={sample.shape[0]} ,label={label.shape[0]} "
        self.length = sample.shape[0]
        self.batch_size = batch_size
        self.nums_batch = int(np.ceil(sample.shape[0] / batch_size))

        if shuffle:
            index = np.random.permutation(self.length)
            self.sample = sample[index]
            self.label = label[index]
        else:
            self.sample = sample
            self.label = label

    def __iter__(self):
        """
        Iterator used for outputting samples and labels.

        :return: Each batch of samples and labels used for training.
        """
        for batch_idx in range(self.nums_batch):
            x = self.sample[self.batch_size * batch_idx:min(self.batch_size * (batch_idx + 1), self.sample.shape[0])]
            y = self.label[self.batch_size * batch_idx:min(self.batch_size * (batch_idx + 1), self.label.shape[0])]
            yield x, y

    def __len__(self):
        """

        :return: The length of the batch.
        """
        return self.nums_batch

    def get_all(self):
        """

        :return: All samples and labels.
        """
        return self.sample, self.label

    def total_length(self):
        """

        :return: the length of all sample.
        """
        return self.length


def MINST_loader(conv=False, batch_size=32):
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
            np.float32)
    with gzip.open(path[2], 'rb') as path3:
        test_labels = np.frombuffer(path3.read(), np.uint8, offset=8)
    with gzip.open(path[3], 'rb') as path4:
        test_images = np.frombuffer(path4.read(), np.uint8, offset=16).reshape((len(test_labels),) + shape).astype(
            np.float32)
    if conv:
        train_images = np.expand_dims(train_images, axis=-1)
        test_images = np.expand_dims(test_images, axis=-1)

    def get_one_hot(labels):
        one_hot = np.zeros((len(labels), 10))
        for i in range(len(labels)):
            one_hot[i, labels[i]] = 1
        return one_hot

    train_one_hot = get_one_hot(train_labels).astype(np.float32)
    test_one_hot = get_one_hot(test_labels).astype(np.float32)

    return Dataset(train_images, train_one_hot, batch_size=batch_size), Dataset(test_images, test_one_hot)
