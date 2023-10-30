import numpy as np
from numpy import ndarray


class Dataset:
    def __init__(self, sample: ndarray, label: ndarray):
        assert sample.shape[0] == label.shape[0], f"数据集的样本数不对应:sample={sample.shape[0]} ,label={label.shape[0]}"
        self.dataset = list(zip(sample, label))

    def __iter__(self):
        for sample, label in self.dataset:
            if len(sample.shape) == 1:
                yield sample.reshape(-1, 1), label
            else:
                yield sample.reshape, label

# todo 写一个处理数据集的东西 处理成numpy格式再传进dataset模块 输出Dataset
# todo 写一个shuffle数据的东西
