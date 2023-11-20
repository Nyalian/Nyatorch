import numpy as np

from activation import Sigmoid, Hebb_act, MLP_act
from layer import LinearLayer
from loss import MeanSquaredError
from sequential import Sequential
from dataset import Dataset
from optimizer import SGD
import matplotlib.pyplot as plt

net = Sequential(
    LinearLayer(2, 2),
    MLP_act(),
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
