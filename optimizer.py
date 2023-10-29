class Optimizer:
    def __init__(self, model, learning_rate):
        self.model = model
        self.learning_rate = learning_rate

    def update(self, gradients):
        pass


class SGD(Optimizer):
    def update(self, gradients):
        for layer, gradient in zip(self.model.layers, gradients):
            layer.weights -= self.learning_rate * gradient[0]
            layer.bias -= self.learning_rate * gradient[1]
