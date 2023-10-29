import numpy as np

from activation import Sigmoid
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

num_epochs = 10000
learning_rate = 0.5
loss = MeanSquaredError()
optimizer = SGD(net, learning_rate)
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

print(f"{[0,0]}={net(np.array([0,0]))>0.5}")
print(f"{[1,0]}={net(np.array([1,0]))>0.5}")
print(f"{[0,1]}={net(np.array([0,1]))>0.5}")
print(f"{[1,1]}={net(np.array([1,1]))>0.5}")