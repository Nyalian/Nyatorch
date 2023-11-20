import numpy as np
from tqdm import tqdm

from dataset import load_MINST

from activation import Sigmoid, ReLU, Linear
from layer import LinearLayer
from loss import CrossEntropy, MeanSquaredError
from sequential import Sequential
from optimizer import SGD

net = Sequential(
    LinearLayer(784, 256),
    Sigmoid(),
    LinearLayer(256, 100),
    ReLU(),
    LinearLayer(100, 64),
    Sigmoid(),
    LinearLayer(64, 10),
    Linear(),
)

train, test = load_MINST()

num_epochs = 20
learning_rate = 0.001
loss = MeanSquaredError()
optimizer = SGD(net, learning_rate)

for epoch in range(num_epochs):
    total_loss = 0

    for inputs, target in tqdm(train):
        # 使用权重更新对象进行训练
        outputs = net(inputs)
        net.backward(loss, target)
        total_loss += loss.calculate(outputs,target)
        optimizer.update()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}, Loss: {total_loss}")

corrects = 0
for inputs, target in test:
    pred = net(inputs)
    pred_idx = np.argmax(pred, axis=0)
    label_idx = np.argmax(target, axis=0)
    corrects += (pred_idx==label_idx)/100

print(corrects)
