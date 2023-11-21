from tqdm import tqdm

from activation import Sigmoid, ReLU, Linear
from dataset import MINST_loader
from indicator import accuracy
from layer import LinearLayer
from loss import CrossEntropy
from optimizer import SGD
from sequential import Sequential

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

train, test = MINST_loader()

num_epochs = 2
learning_rate = 0.003
loss = CrossEntropy()
net.def_loss(loss)
optimizer = SGD(net, learning_rate)

for epoch in range(num_epochs):
    total_loss = 0

    for inputs, target in tqdm(train):
        # 使用权重更新对象进行训练
        outputs = net(inputs)
        net.backward(target)
        total_loss += loss.calculate(outputs, target)
        optimizer.update()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}, Loss: {total_loss}")

net.save_module()

net.load_module()
test_x, test_y = test.get_all()
corrects = accuracy(net(test_x), test_y)
print(corrects)
