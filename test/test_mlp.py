import numpy as np

from Nyatorch import nn


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
    [1,0],
    [0,1],
    [0,1]
])

num_epoch = 20

for i in range(num_epoch):
    a = net(x.T)
    net.mlp_func(x.T, a, y.T)

print(net(x.T).T)
