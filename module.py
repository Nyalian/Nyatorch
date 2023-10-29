from typing import Iterator, Any, List


class Module:

    def __init__(self):
        self._modules: List['Module'] = list()
        self.layer_outputs = []

    def __call__(self, *input, **kwargs):
        return self.forward(*input)

    def forward(self, *input: Any):
        raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required \"forward\" function")

    def add_module(self, module: 'Module') -> None:
        self._modules.append(module)


class Sequential(Module):

    def __init__(self, *args: Module):
        super(Sequential, self).__init__()
        for module in args:
            self.add_module(module)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules)

    def forward(self, input):
        output = input
        self.layer_outputs = [output]  # 存储输入层的输出
        for module in self:
            output = module.forward(output)
            self.layer_outputs.append(output)  # 存储每一层的输出
        return output
