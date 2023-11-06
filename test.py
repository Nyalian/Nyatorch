import numpy as np

from activation import Sigmoid
from bp import Backpropagation
from layer import LinearLayer
from loss import MeanSquaredError
from sequential import Sequential
from dataset import Dataset
from optimizer import SGD

net = Sequential(
    LinearLayer(2, 4),
    Sigmoid(),
    LinearLayer(4, 1),
    Sigmoid(),
)

x = np.array([[0, 1], [0, 0], [1, 0], [1, 1]])
y = np.array([[1], [0], [1], [0]])
data = Dataset(x, y)

num_epochs = 1000
learning_rate = 0.5
loss = MeanSquaredError()
optimizer = SGD(net, learning_rate)
bp = Backpropagation(net, loss)
for epoch in range(num_epochs):
    total_loss = 0

    for inputs, target in data:
        # 使用权重更新对象进行训练
        outputs = net(inputs)
        net.backward(loss, target)
        total_loss += loss.calculate(outputs,target)
        optimizer.update()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, Loss: {total_loss}")

print(net(np.array([0,0])))
