from typing import Iterator, Any, List

from numpy import ndarray

from layer import Layer
from loss import Loss


class Module:

    def __init__(self):
        pass

    def __call__(self, *input, **kwargs):
        return self.forward(*input)

    def forward(self, *input: Any):
        raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required \"forward\" function")

    def backward(self, *input:Any):
        #todo
        raise NotImplementedError()


class Sequential(Module):

    def __init__(self, *args: Module):
        self._modules: List['Module'] = list()
        self.layer_outputs = []
        self.delta = []
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
        output = loss.gradient(self.layer_outputs[-1], target)
        for module, para in self.get_zip():
            if isinstance(self.module, Layer):
                pass
