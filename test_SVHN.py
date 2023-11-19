import numpy as np
from tqdm import tqdm

from dataset import load_MINST

from activation import Sigmoid, ReLU, Linear
from layer import LinearLayer, Conv2d, Flatten
from loss import CrossEntropy, MeanSquaredError
from sequential import Sequential
from optimizer import SGD
import scipy.io

train_data = scipy.io.loadmat('train_32x32.mat')
train_x = train_data['X']
train_y = train_data["y"]

test_data = scipy.io.loadmat('test_32x32.mat')
test_x = test_data['X']
test_y = test_data["y"]

train_x = train_x.transpose(3, 2, 0, 1)
test_x = test_x.transpose(3, 2, 0, 1)
print(train_y.shape)

net = Sequential(
    Conv2d(3, 10, 3),
    Sigmoid(),
    Conv2d(10, 20, 3),
    ReLU(),
    Flatten(),
    LinearLayer(15680, 10),
)

num_epochs = 20
learning_rate = 0.01
loss = MeanSquaredError()
optimizer = SGD(net, learning_rate)

train = zip(train_x[:100], train_y[:100])

for epoch in range(num_epochs):
    total_loss = 0

    for inputs, target in tqdm(train):
        # 使用权重更新对象进行训练
        outputs = net(inputs)
        net.backward(loss, target)
        total_loss += loss.calculate(outputs, target)
        print(inputs.shape)
        optimizer.update()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}, Loss: {total_loss}")
