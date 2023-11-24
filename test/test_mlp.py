import numpy as np

from Nyatorch import nn
from Nyatorch.utils.data import DataLoader

net = nn.Sequential(
    nn.LinearLayer(2, 2),
    nn.MLPAct(),
)

x = np.array([
    [1, 2],
    [-1, 2],
    [0, -1]
])

y = np.array([
    [1, 0],
    [0, 1],
    [0, 1]
])

num_epoch = 20

train = DataLoader(x, y)

for i in range(num_epoch):
    for x, y in train:
        a = net(x)
        net.mlp_func(x, a, y)

for x, y in train:
    print(net(x))
