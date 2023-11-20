import numpy as np
from tqdm import tqdm

from dataset import load_MINST, load_MINST_conv

from activation import Sigmoid, ReLU, Linear
from layer import LinearLayer, Conv2d, Flatten
from loss import CrossEntropy, MeanSquaredError
from sequential import Sequential
from optimizer import SGD
import scipy.io

train, test = load_MINST_conv()

net = Sequential(
    Conv2d(1, 1, 3),
    Sigmoid(),
    Conv2d(1, 1, 3),
    ReLU(),
    Flatten(),
    LinearLayer(576, 10),
)

num_epochs = 5
learning_rate = 0.02
loss = MeanSquaredError()
optimizer = SGD(net, learning_rate)

for epoch in range(num_epochs):
    total_loss = 0

    for inputs, target in tqdm(train):
        # 使用权重更新对象进行训练
        outputs = net(inputs)
        net.backward(loss, target)
        total_loss += loss.calculate(outputs, target)
        # print(inputs.shape)
        optimizer.update()
    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch + 1}, Loss: {total_loss}")

corrects = 0
for inputs, target in test:
    pred = net(inputs)
    pred_idx = np.argmax(pred, axis=0)
    label_idx = np.argmax(target, axis=0)
    corrects += (pred_idx == label_idx) / test.total_length()

print(corrects)
