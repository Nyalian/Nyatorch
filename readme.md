# 介绍

--- 
Nyatorch 是一个神经网络框架，提供了最基础的全连接层和卷积网络框架  
主要特点如下：

- 全连接网络
- 卷积网络（GPU和CPU）
- 自定义神经元个数
- 自定义层数
- 数据预处理
- 打乱数据集(shuffle)
- 分批样本(batch_size)
- 多种学习算法
- 两种以上性能函数(损失函数)
- 神经网络训练和预测
- minst数据集预处理
- minst数据集应用

# requirements

---

- numpy
- numba （如果使用GPU版卷积网络）
- cuda 核心（如果使用GPU版卷积网络）

# 结构

---

## Nyatorch.utils

### class MeanSquaredError()

MSE损失函数，用于计算损失和反向传播更新

#### calculate(prediction, label)

计算损失

参数说明：

- prediction(ndarray) - 预测值
- label(ndarray) - 标签值

### class CrossEntropy()

交叉熵损失函数，用于计算损失和反向传播更新

#### calculate(prediction, label)

计算损失

参数说明：

- prediction(ndarray) - 预测值
- label(ndarray) - 标签值

### class SGD(net, learning_rate)

梯度下降算法，用于更新权重。

参数说明：

- net(Sequential) - 网络结构
- learning_rate(float) - 学习率

#### update()

在反向传播完之后，更新权重。

```python
import numpy as np
from Nyatorch import nn

from Nyatorch.utils import MeanSquaredError, SGD
from Nyatorch.utils.data import DataLoader

net = nn.Sequential(
    nn.LinearLayer(2, 4),
    nn.Sigmoid(),
    nn.LinearLayer(4, 1),
    nn.Sigmoid(),
)

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
data = DataLoader(x, y)

num_epochs = 10000
learning_rate = 0.5
loss = MeanSquaredError()
net.def_loss(loss)

# 定义优化器
optimizer = SGD(net, learning_rate)

for epoch in range(num_epochs):
    total_loss = 0

    for inputs, target in data:
        outputs = net(inputs)
        net.backward(target)
        # 计算损失值
        total_loss += loss.calculate(outputs, target)
        # 使用权重更新对象进行训练
        optimizer.update()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, Loss: {total_loss}")

```

### func accuracy(predict, label)

计算并返回准确率

参数说明：

- prediction(ndarray) 预测值。应该传入的是一个one_hot编码
- label(ndarray) 标签值。应该传入的是一个one_hot编码

## Nyatorch.utils.data

### class DataLoader(sample, label, batch_size=1, shuffle=True)

用来处理数据集的类,你的样本应该通过`DataLoader`类来构造。

参数说明：

- sample(ndarray) - 样本。你传入的格式应该是 `[size, features]` 或者 `[size, width, height, channel]`。
- label(ndarray) - 标签。你传入的格式应该是 `[size, features]` 并且和`sample`的`size`相等。
- batch_size(int, optional) - 用来每一次模型更新时所使用的样本数量。默认为`1`。
- shuffle(bool, optional) - 如果为`True`，则打乱传入的数据集。默认为`True`。

```python
from Nyatorch.utils.data import DataLoader
import numpy as np

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
train = DataLoader(x, y)
```

#### get_all()

返回所有的样本和标签。

#### total_length()

返回样本的数量。  
**注意：** 与`len()`不同，`len()`返回的是`num_batch`，即批次的总数。

### func MINST_loader(conv=False, batch_size=32)

返回装载MINST数据集的训练`DataLoader`和测试`DataLoader`

参数说明：

- conv(bool, optional) - 如果为True，将按`[size, width, height, channel]`的形状来构造数据集，否则将按`[size, features]`的格式来构造。
- batch_size(int, optional) - 用来每一次模型更新时所使用的样本数量。默认为`32`。

```python
from Nyatorch.utils.data import MINST_loader

train, test = MINST_loader(conv=True, batch_size=512)
```

## Nyatorch.nn

---
**连接层、卷积层、池化层、展平**

### class LinearLayer(in_features, out_features)

全连接层，用于全连接的计算。

参数说明：

- in_features(int) - 前一层的神经元数量
- out_features(int) - 后一层的神经元数量

主要成员：

- weights(ndarray) - 权重。形状为`[in_features, out_features]`
- bias(ndarray) - 偏置。形状为`[1, out_features]`
- gradient_weights(ndarray) - 权重的梯度。形状为`[in_features, out_features]`
- gradient_bias(ndarray) - 偏置的梯度。形状为`[1, out_features]`

### class Conv2d(in_channel, out_channel, kernel_size, padding, stride)

二维卷积层的CPU版，用于卷积的计算。

参数说明：

- in_channel(int) - 输入通道
- out_channel(int) - 输出通道
- padding(int) - 填充
- stride(int) - 步长

主要成员：

- weights(ndarray) - 权重。形状为`[kernel_size, kernel_size, in_channel, out_channel]`
- bias(ndarray) - 偏置。形状为`[1, out_channel]`
- gradient_weights(ndarray) - 权重的梯度。形状为`[kernel_size, kernel_size, in_channel, out_channel]`
- gradient_bias(ndarray) - 偏置的梯度。形状为`[1, out_channel]`

### class Conv2d_GPU(in_channel, out_channel, kernel_size, padding, stride)

二维卷积层的GPU版，用于卷积的计算。

参数和主要成员和CPU版相同。

### class MaxPooling(pool_size)

最大池化层的CPU版，用于池化层的计算。

参数说明：

- pool_size(int) - 池化层大小

### class MaxPoolingGPU(pool_size)

最大池化层的GPU版，用于池化层的计算。

参数和CPU版相同。

### class Flatten()

展平层。用于卷积层向全连接层的展平。

---
**激活函数层**

### class Linear()

线性激活函数层

### class ReLU()

ReLU激活函数层

### class Sigmoid()

Sigmoid激活函数层

### class Tanh()

Tanh激活函数层

### class HebbAct()

Hebb算法，和`Sequential.hebb()`配合使用

### class MLPAct()

MLP算法，和`Sequential.mlp_func()`一起使用

---
**Sequential容器**

### Sequential(*args, loss=None)

用于构建每一层的连接层。你的网络模型的构建应该通过`Sequential`来实现。

参数说明：

- *args(Module) - 传入的层。`Module`是所有层的父类，所以传入任何层都是可行的。（除了`HebbAct`和`MLPAct`）。
- loss(Loss) - 损失函数。如果未指定，则在调用`backward`必须通过`def_loss()`来指定。

```python
from Nyatorch import nn
from Nyatorch.utils import MeanSquaredError

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

loss = MeanSquaredError()
net.def_loss(loss)
```

#### def_loss(loss)

传入损失函数。如果在创建`Sequential`实例时没有定义`loss`，则在调用`backward`前需要传入。

参数说明：

- loss(Loss) - 损失函数。

#### forward(inputs)

容器的前向传播，并且获得输出值。

参数说明：

- inputs(ndarray) - 样本。你的样本输入应该通过`DataLoader`的迭代器来获得并输入。

#### backward(label)

容器的反向传播，并记录权重梯度，用来优化器的权重更新。

参数说明：

- label(ndarray) - 标签。你的标签输入应该通过`DataLoader`的迭代器来获得并输入。

#### save_module(file_name)

保存模型。

参数说明：

- file_name(path) - 文件名或路径。

#### save_module(file_name)

加载模型。请确保你的模型和`file_name`的模型对应。

参数说明：

- file_name(path) - 文件名或路径。

#### hebb(input, label, learning_rate)

用hebb算法来更新权重。使用`hebb`时，确保你的模型只有两层：`LinearLayer`和`HebbAct`

参数说明：

- input(ndarray) - 样本。
- input(ndarray) - 标签。
- learning_rate - 学习率。

```python
from Nyatorch import nn
from Nyatorch.utils.data import DataLoader

# 构建数据集
train = DataLoader(x, t)

# 构建网络
net = nn.Sequential(
    nn.LinearLayer(30, 30),
    nn.HebbAct(),
)
learning_rate = 0.5

for input, target in train:
    # 更新权重
    net.hebb(input, target, learning_rate)
```

#### mlp_func(input, output, label)

用mlp算法来更新权重。使用`hebb`时，确保你的模型只有两层：`LinearLayer`和`MLPAct`

参数说明：

- input(ndarray) - 样本。
- output(ndarray) - 输出。
- learning_rate - 学习率。

```python
from Nyatorch import nn
from Nyatorch.utils.data import DataLoader

# 构建数据集
train = DataLoader(x, t)

# 构建网络
net = nn.Sequential(
    nn.LinearLayer(2, 2),
    nn.MLPAct(),
)

for x, y in train:
    a = net(x)
    # 更新权重
    net.mlp_func(x, a, y)
```

# 使用案例

以下是通过MINST数据集进行计算的样例：

```python
from tqdm import tqdm

from Nyatorch import nn
from Nyatorch.utils import CrossEntropy, SGD, accuracy
from Nyatorch.utils.data import MINST_loader

# 获取训练集和测试集合
train, test = MINST_loader(conv=True, batch_size=64)

# 定义网络
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

# 循环次数和学习率
num_epochs = 10
learning_rate = 0.001

# 定义损失函数，传入损失函数，定义优化器
loss = CrossEntropy()
net.def_loss(loss)
optimizer = SGD(net, learning_rate)

for epoch in range(num_epochs):

    total_loss = 0
    for inputs, label in tqdm(train):
        # 前向传播获得输出值
        outputs = net(inputs)
        # 反向传播
        net.backward(label)
        # 计算损失
        total_loss += loss.calculate(outputs, label) / len(train)
        # 更新权重
        optimizer.update()

    if (epoch + 1) % 2 == 0:
        print(f"Epoch {epoch + 1}, Loss: {total_loss}")

# 获取测试集的所有样本
# 如果这一步的计算报错的话（因为显存过小），可以和训练一样取batch然后计算准确率，或者使用CPU版进行计算
test_x, test_y = test.get_all()
corrects = accuracy(net(test_x), test_y)
print(corrects)
```