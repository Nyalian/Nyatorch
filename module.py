from typing import Any


class Module:

    def __init__(self):
        pass

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs)

    def forward(self, *inputs: Any):
        raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required \"forward\" function")

    def backward(self, *delta: Any):
        raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required \"backward\" function")

    def get_parameter(self):
        return {}

    def set_parameter(self, parameter: dict):
        pass
