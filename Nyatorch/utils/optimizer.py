from Nyatorch import nn


class Optimizer:
    def __init__(self, net: 'nn.Sequential', learning_rate: float) -> None:
        self.net = net
        self.learning_rate = learning_rate

    def update(self) -> None:
        pass


class SGD(Optimizer):
    def update(self) -> None:
        for module in self.net:
            module.update(self.learning_rate)
