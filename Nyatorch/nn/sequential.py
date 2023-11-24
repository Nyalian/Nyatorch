from typing import Iterator, List, Optional

import numpy as np
from numpy import ndarray

from .abstract import Module
from .linear import LinearLayer
from Nyatorch.utils.loss import Loss


class Sequential(Module):

    def __init__(self, *args: Module, loss: Optional[Loss] = None) -> None:
        super().__init__()
        self._modules: List['Module'] = list()
        self.outputs = None
        self.loss = loss
        for module in args:
            self.add_module(module)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules)

    def __reversed__(self):
        return reversed(self._modules)

    def def_loss(self, loss: Loss):
        self.loss = loss

    def add_module(self, module: 'Module') -> None:
        self._modules.append(module)

    def forward(self, inputs):
        output = inputs

        for module in self:
            output = module.forward(output)
        self.outputs = output
        return output

    def backward(self, target: ndarray):
        delta = self.loss.gradient(self.outputs, target)
        for module in reversed(self):
            delta = module.backward(delta)

    def save_module(self, file_name: str = 'model.npy'):
        module_list = []
        for module in self:
            module_list.append(module.get_parameter())
        np.save(file_name, module_list)

    def load_module(self, file_name: str = 'model.npy'):
        module_list = np.load(file_name, allow_pickle=True)
        for module, para in zip(self, module_list):
            module.set_parameter(para)

    def hebb(self, input: ndarray, target: ndarray, learning_rate):
        """

        :param learning_rate:
        :param input: [batch_size, train_x_feature]
        :param target: [batch, train_y_feature]
        """
        for module in self._modules:
            if isinstance(module, LinearLayer):
                module.weights += learning_rate * (input @ target.T)

    def mlp_func(self, input: ndarray, output: ndarray, target: ndarray):
        """

        :param input: [batch_size, train_x_feature]
        :param output: [batch_size, out_feature]
        :param target: [batch_size, train_y_feature]
        """
        for module in self._modules:
            if isinstance(module, LinearLayer):
                module.weights += input.T @ (target - output)
                module.bias += (target - output).sum(axis=0, keepdims=True)
