import numpy as np
from numpy import ndarray
import gzip


class Dataset:
    def __init__(self, sample: ndarray, label: ndarray, batch_size=1, shuffle=True, is_transpose=True):
        assert sample.shape[0] == label.shape[0], f"数据集的样本数不对应:sample={sample.shape[0]} ,label={label.shape[0]}"
        self.length = sample.shape[0]
        self.batch_size = batch_size
        self.nums_batch = int(np.ceil(sample.shape[0] / batch_size))
        self.is_transpose = is_transpose

        if shuffle:
            index = np.random.permutation(self.length)
            self.sample = sample[index]
            self.label = label[index]
        else:
            self.sample = sample
            self.label = label

    def __iter__(self):
        '''

        :return:
        '''
        for batch_idx in range(self.nums_batch):
            x = self.sample[self.batch_size * batch_idx:min(self.batch_size * (batch_idx + 1), self.sample.shape[0])]
            y = self.label[self.batch_size * batch_idx:min(self.batch_size * (batch_idx + 1), self.label.shape[0])]
            # todo

            if self.is_transpose:
                yield (x.T, y.T)
            else:
                yield (x, y.T)

    def __len__(self):
        return self.nums_batch

    def total_length(self):
        return self.length


def load_MINST():
    path0 = './data/MNIST/raw/'
    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']

    path = [path0 + each for each in files]

    with gzip.open(path[0], 'rb') as lbpath:
        train_labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(path[1], 'rb') as imgpath:
        train_images = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(train_labels), 784)
    with gzip.open(path[2], 'rb') as lbpath:
        test_labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(path[3], 'rb') as imgpath:
        test_images = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(test_labels), 784)

    train_onehot = np.zeros((len(train_labels), 10))
    for i in range(len(train_labels)):
        train_onehot[i, train_labels[i]] = 1

    test_onehot = np.zeros((len(test_labels), 10))
    for i in range(len(test_labels)):
        test_onehot[i, test_labels[i]] = 1

    return Dataset(train_images, train_onehot, batch_size=32), Dataset(test_images, test_onehot)


def load_MINST_conv():
    path0 = './data/MNIST/raw/'
    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']

    path = [path0 + each for each in files]

    with gzip.open(path[0], 'rb') as lbpath:
        train_labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(path[1], 'rb') as imgpath:
        train_images = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(train_labels), 28, 28)
        # train_images = np.expand_dims(train_images, axis=1)
    with gzip.open(path[2], 'rb') as lbpath:
        test_labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(path[3], 'rb') as imgpath:
        test_images = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(test_labels), 28, 28)
        # test_images = np.expand_dims(test_images, axis=1)

    train_onehot = np.zeros((len(train_labels), 10))
    for i in range(len(train_labels)):
        train_onehot[i, train_labels[i]] = 1

    test_onehot = np.zeros((len(test_labels), 10))
    for i in range(len(test_labels)):
        test_onehot[i, test_labels[i]] = 1

    return Dataset(train_images, train_onehot, batch_size=1, is_transpose=False), Dataset(test_images, test_onehot,
                                                                                          is_transpose=False)
