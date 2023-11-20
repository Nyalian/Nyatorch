import numpy as np
from numpy import ndarray
import gzip


class Dataset:
    def __init__(self, sample: ndarray, label: ndarray, batch_size=1, is_transpose=True, is_crop=False):
        assert sample.shape[0] == label.shape[0], f"数据集的样本数不对应:sample={sample.shape[0]} ,label={label.shape[0]}"
        self.sample = sample
        self.label = label
        self.batch_size = batch_size
        self.nums_batch = int(np.ceil(sample.shape[0] / batch_size))
        self.is_transpose = is_transpose
        self.is_crop = is_crop

    def __iter__(self):
        '''

        :return:
        '''
        for batch_idx in range(self.nums_batch):
            x = self.sample[self.batch_size * batch_idx:min(self.batch_size * (batch_idx + 1), self.sample.shape[0])]
            y = self.label[self.batch_size * batch_idx:min(self.batch_size * (batch_idx + 1), self.label.shape[0])]
            #todo

            if self.is_transpose:
                yield (x.T, y.T)
            else:
                yield (x, y.T)
            '''for sample, label in self.dataset:
                        if len(sample.shape) == 1:
                            sample = sample.reshape(-1, 1)
                        if len(label.shape) == 1:
                            label = label.reshape(-1, 1)
                        yield sample, label'''

    def __len__(self):
        return len(self.nums_batch)


# todo 写一个shuffle数据的东西


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

    return Dataset(train_images, train_onehot, batch_size=1, is_transpose=False), Dataset(test_images, test_onehot, is_transpose=False)
