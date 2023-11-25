from typing import Optional, Tuple
from numpy import ndarray

import numpy as np


class DataLoader:
    def __init__(self, sample: ndarray, label: ndarray, batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = True) -> None:
        """
        Initialize the incoming data and perform other operations.

        :param sample: The sample data. It should be [size, features] or [size, width, height, channel].
        :param label: The label data. It should be [size, features]
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

    def __iter__(self) -> Tuple[ndarray, ndarray]:
        """
        Iterator used for outputting samples and labels.

        :return: Each batch of samples and labels used for training or testing.
        """
        for batch_idx in range(self.nums_batch):
            x = self.sample[self.batch_size * batch_idx:min(self.batch_size * (batch_idx + 1), self.sample.shape[0])]
            y = self.label[self.batch_size * batch_idx:min(self.batch_size * (batch_idx + 1), self.label.shape[0])]
            yield x, y

    def __len__(self) -> int:
        """

        :return: The length of the batch.
        """
        return self.nums_batch

    def get_all(self) -> Tuple[ndarray, ndarray]:
        """

        :return: All samples and labels.
        """
        return self.sample, self.label

    def total_length(self) -> int:
        """

        :return: the length of all sample.
        """
        return self.length
