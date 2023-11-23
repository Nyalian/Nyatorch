import numpy as np
from tqdm import tqdm

from activation import ReLU, Sigmoid
from dataset import MINST_loader
from indicator import accuracy
from layer_cuda import LinearLayer, Conv2d, Flatten, MaxPooling
from loss import CrossEntropy
from optimizer import SGD
from sequential import Sequential

train, test = MINST_loader(conv=True, batch_size=64)
net = Sequential(
    Conv2d(1, 4, 3),
    ReLU(),
    Conv2d(4, 8, 3),
    ReLU(),
    MaxPooling(2),
    Flatten(),
    LinearLayer(1152, 512),
    Sigmoid(),
    LinearLayer(512, 10),
)

num_epochs = 10
learning_rate = 0.001
loss = CrossEntropy()
net.def_loss(loss)
optimizer = SGD(net, learning_rate)

for epoch in range(num_epochs):
    total_loss = 0

    for inputs, target in tqdm(train):
        # 使用权重更新对象进行训练
        outputs = net(inputs)
        net.backward(target)
        total_loss += loss.calculate(outputs, target) / len(train)
        optimizer.update()

    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch + 1}, Loss: {total_loss}")


test_x, test_y = test.get_all()
corrects = accuracy(net(test_x), test_y)
print(corrects)
