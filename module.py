from typing import Any


class Module:

    def __init__(self):
        pass

    def __call__(self, *input, **kwargs):
        return self.forward(*input)

    def forward(self, *input: Any):
        raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required \"forward\" function")

    def backward(self, *input: Any):
        raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required \"backward\" function")
