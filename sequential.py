from numpy import ndarray
from typing import Iterator, List
from activation import Activation
from layer import LinearLayer
from loss import Loss
from module import Module


class Sequential(Module):

    def __init__(self, *args: Module):
        self._modules: List['Module'] = list()
        self.layer_outputs = []
        super(Sequential, self).__init__()
        for module in args:
            self.add_module(module)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules)

    def add_module(self, module: 'Module') -> None:
        self._modules.append(module)

    def get_zip(self):
        return reversed(list(zip(self._modules, self.layer_outputs)))

    def forward(self, input):
        output = input
        self.layer_outputs = [output]  # 存储输入层的输出
        for module in self:
            output = module.forward(output)
            self.layer_outputs.append(output)  # 存储每一层的输出
        return output

    def backward(self, loss: Loss, target: ndarray):
        delta = loss.gradient(self.layer_outputs[-1], target)
        for module, para in self.get_zip():
            if isinstance(module, LinearLayer):
                module.gradient_weights = delta @ para.T
                module.gradient_bias = delta.sum(axis=1, keepdims=True)
                delta = module.weights.T @ delta
            if isinstance(module, Activation):
                delta = module.backward(para) * delta

    def hebb(self, input: ndarray, target:ndarray, learning_rate):
        for module in self._modules:
            if isinstance(module , LinearLayer):
                module.weights += learning_rate*(target @ input.T)

    def mlp_func(self,input: ndarray,output: ndarray, target:ndarray):
        for module in self._modules:
            if isinstance(module , LinearLayer):
                pass
            #todo



