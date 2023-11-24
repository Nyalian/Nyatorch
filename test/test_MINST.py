from tqdm import tqdm

from Nyatorch import nn
from Nyatorch.utils import CrossEntropy, SGD, accuracy
from Nyatorch.utils.data import MINST_loader

net = nn.Sequential(
    nn.LinearLayer(784, 256),
    nn.Sigmoid(),
    nn.LinearLayer(256, 100),
    nn.ReLU(),
    nn.LinearLayer(100, 64),
    nn.Sigmoid(),
    nn.LinearLayer(64, 10),
    nn.Linear(),
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
        total_loss += loss.calculate(outputs, target) / len(train)
        optimizer.update()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}, Loss: {total_loss}")

net.save_module()

net.load_module()
test_x, test_y = test.get_all()
corrects = accuracy(net(test_x), test_y)
print(corrects)
