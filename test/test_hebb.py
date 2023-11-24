import numpy as np

from Nyatorch import nn
import matplotlib.pyplot as plt


def show(x):
    y = [1 if i > 0 else 0 for i in x]
    y = np.array(y).reshape(6, 5)

    # 使用imshow函数显示二值数组
    plt.imshow(y, cmap='binary')

    # 添加颜色条以显示对应的值
    plt.colorbar()

    # 显示图形
    plt.show()


net = nn.Sequential(
    nn.LinearLayer(30, 30),
    nn.HebbAct(),
)

# 输入值
x = np.array([[-1, 1, 1, 1, -1,
               1, -1, -1, -1, 1,
               1, -1, -1, -1, 1,
               1, -1, -1, -1, 1,
               1, -1, -1, -1, 1,
               -1, 1, 1, 1, -1],
              [-1, 1, 1, -1, -1,
               -1, -1, 1, -1, -1,
               -1, -1, 1, -1, -1,
               -1, -1, 1, -1, -1,
               -1, -1, 1, -1, -1,
               -1, -1, 1, -1, -1, ],
              [1, 1, 1, -1, -1,
               -1, -1, -1, 1, -1,
               -1, -1, -1, 1, -1,
               -1, 1, 1, -1, -1,
               -1, 1, -1, -1, -1,
               -1, 1, 1, 1, 1]])

# 输出值
t = np.array([[-1, 1, 1, 1, -1,
               1, -1, -1, -1, 1,
               1, -1, -1, -1, 1,
               1, -1, -1, -1, 1,
               1, -1, -1, -1, 1,
               -1, 1, 1, 1, -1],
              [-1, 1, 1, -1, -1,
               -1, -1, 1, -1, -1,
               -1, -1, 1, -1, -1,
               -1, -1, 1, -1, -1,
               -1, -1, 1, -1, -1,
               -1, -1, 1, -1, -1, ],
              [1, 1, 1, -1, -1,
               -1, -1, -1, 1, -1,
               -1, -1, -1, 1, -1,
               -1, 1, 1, -1, -1,
               -1, 1, -1, -1, -1,
               -1, 1, 1, 1, 1]])

num_epoch = 1
learning_rate = 0.5
for i in range(num_epoch):
    for input, target in zip(x, t):
        net.hebb(input.reshape(1, -1), target.reshape(1, -1), learning_rate)

for a in x:
    a = a.reshape(-1, 1)
    show(net(a))
