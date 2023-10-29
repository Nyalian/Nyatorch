# 可能已经弃用
class Backpropagation:
    def __init__(self, model, loss):
        self.model = model
        self.loss = loss

    #todo backward现在还跑不了 得改一下 遇到activation层要跳过并且记入 用到上一层的导数上
    def backward(self, target, outputs):
        layer_outputs = self.model.layer_outputs
        deltas = []

        # 计算损失
        prediction = layer_outputs[-1]
        loss_value = self.loss.calculate(prediction, target)

        # 计算输出层的 delta
        delta = self.loss.gradient(prediction, target) * prediction * (1 - prediction)
        deltas.append(delta)

        # 计算隐藏层的 delta
        for i in range(len(self.model._modules) - 2, 0, -1):
            weights_next_layer = self.model._modules[i + 1].weights
            error = deltas[-1].dot(weights_next_layer.T)
            delta = error * layer_outputs[i] * (1 - layer_outputs[i])
            deltas.append(delta)

        deltas.reverse()

        # 计算梯度
        gradients = []
        for i in range(len(self.model.layers) - 1):
            layer = self.model.layers[i]
            gradient = [
                layer_outputs[i].reshape(-1, 1) @ deltas[i].reshape(1, -1),  # 权重的梯度
                deltas[i]  # 偏置的梯度
            ]
            gradients.append(gradient)

        return loss_value, gradients