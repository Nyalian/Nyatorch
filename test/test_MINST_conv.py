from tqdm import tqdm

from Nyatorch import nn
from Nyatorch.utils import CrossEntropy, SGD, accuracy
from Nyatorch.utils.data import MINST_loader

train, test = MINST_loader(conv=True, batch_size=64)
net = nn.Sequential(
    nn.Conv2dGPU(1, 4, 3),
    nn.ReLU(),
    nn.Conv2dGPU(4, 8, 3),
    nn.ReLU(),
    nn.MaxPoolingGPU(2),
    nn.Flatten(),
    nn.LinearLayer(1152, 512),
    nn.Sigmoid(),
    nn.LinearLayer(512, 10),
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
