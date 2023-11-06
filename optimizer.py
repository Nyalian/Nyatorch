from layer import LinearLayer


class Optimizer:
    def __init__(self, net, learning_rate) -> None:
        self.net = net
        self.learning_rate = learning_rate

    def update(self, gradients) -> None:
        pass

class SGD(Optimizer):
    def update(self) -> None:
        for module in self.net:
            if isinstance(module, LinearLayer):
                module.weights -= self.learning_rate * module.gradient_weights
                module.bias -= self.learning_rate * module.gradient_bias


