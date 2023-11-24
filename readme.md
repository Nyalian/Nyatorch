# 介绍
# **可能考虑重写**

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
- 两种以上性能函数
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
## class Dataset(sample, label, batch_size=1, shuffle=True)
用来处理数据集的类,你的样本应该通过`Dataset`类来构造。

参数说明：
- sample(ndarray) - 样本。你传入的格式应该是 `[size, features]` 或者 `[size, width, height, channel]`。
- label(ndarray) - 标签。你传入的格式应该是 `[size, features]` 并且和`sample`的`size`相等。
- batch_size(int, optional) - 用来每一次模型更新时所使用的样本数量。默认为`1`。
- shuffle(bool, optional) - 如果为`True`，则打乱传入的数据集。默认为`True`。

```python
from Nyatorch.utils.data.loader import DataLoader
import numpy as np

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
train = DataLoader(x, y)
```
### get_all()  
返回所有的样本和标签。

### total_length()
返回样本的数量。  
**注意：** 与`len()`不同，`len()`返回的是`num_batch`，即批次的总数。

## class Module
`Module` 作为一个抽象类，也作为所有层（包括`Sequential`）的父类  
`LinearLayer`、`Activation`、`ConvNd`、`MaxPooling`、`MaxPoolingGPU`、`Sequential`都继承于此层。

## class LinearLayer(in_features, out_features)
全连接层，用于全连接的计算。  

主要成员:  
- weights(ndarray) - 权重。形状为`[in_features, out_features]`
- bias(ndarray) - 偏置。形状为`[1, out_features]`
- gradient_weights(ndarray) - 权重的梯度。形状为`[in_features, out_features]`
- gradient_bias(ndarray) - 偏置的梯度。形状为`[1, out_features]`

参数说明：
- in_features(int) - 前一层的神经元数量
- out_features(int) - 后一层的神经元数量  

## class ConvNd(in_channel, out_channel, kernel_size, padding, stride)  
所有卷积层的父类。  

参数说明:
- in_channel(int) - 输入通道
- out_channel(int) - 输出通道
- padding(int) - 填充
- stride(int) - 步长