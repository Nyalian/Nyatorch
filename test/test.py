import numpy as np
from Nyatorch import nn

from Nyatorch.utils import MeanSquaredError, SGD
from Nyatorch.utils.data import DataLoader


net = nn.Sequential(
    nn.LinearLayer(2, 4),
    nn.Sigmoid(),
    nn.LinearLayer(4, 1),
    nn.Sigmoid(),
)

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
data = DataLoader(x, y)

num_epochs = 10000
learning_rate = 0.5
loss = MeanSquaredError()
net.def_loss(loss)

optimizer = SGD(net, learning_rate)
for epoch in range(num_epochs):
    total_loss = 0

    for inputs, target in data:
        # 使用权重更新对象进行训练
        outputs = net(inputs)
        net.backward(target)
        total_loss += loss.calculate(outputs,target)
        optimizer.update()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, Loss: {total_loss}")

for inputs, target in data:
    print(f"{inputs}={net(inputs)>0.5}")