import numba as numba
import numpy as np

from module import Module
from numpy import ndarray
import torch.nn as nn


class LinearLayer(Module):

    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features
        bound = np.sqrt(6. / (self.in_features + self.out_features))
        self.weights = np.random.uniform(-bound, bound, (self.out_features, self.in_features))
        self.bias = np.random.rand(out_features, 1)
        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias = np.zeros_like(self.bias)

    def forward(self, input: ndarray) -> ndarray:
        return self.weights @ input + self.bias

    def backward(self, input: ndarray) -> ndarray:
        return input @ self.weights


class ConvNd(Module):

    def __init__(self, in_channel: int, out_channel: int, kernel_size: int, padding: int, stride: int) -> None:
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.weights = None
        self.bias = None
        self.gradient_weights = None
        self.gradient_bias = None

    def gradient_cal(self, para: ndarray, delta: ndarray) -> ndarray:
        pass

    def conv_mul(self, input: ndarray, kernel: ndarray) -> ndarray:
        pass

    def conv_mul_bp(self, input: ndarray, kernel: ndarray) -> ndarray:
        pass

    def forward(self, input: ndarray) -> ndarray:
        pass

    def backward(self, input: ndarray) -> ndarray:
        pass


class Conv2d(ConvNd):

    def __init__(self, in_channel: int, out_channel: int, kernel_size: int, padding: int = 0, stride: int = 1) -> None:
        super().__init__(in_channel, out_channel, kernel_size, padding, stride)
        bound = np.sqrt(6. / (self.in_channel + self.out_channel))
        self.weights = np.random.uniform(-bound, bound, (self.out_channel, self.in_channel, kernel_size, kernel_size))
        self.gradient_weights = np.zeros_like(self.weights)
        self.bias = np.random.rand(self.out_channel)
        self.gradient_bias = np.zeros_like(self.bias)

    def conv_mul(self, input: ndarray, kernel: ndarray, bias: int) -> ndarray:
        output = np.zeros(((input.shape[0] - self.kernel_size) // self.stride + 1,
                           (input.shape[1] - self.kernel_size) // self.stride + 1))
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i][j] = (input[i * self.stride:i * self.stride + self.kernel_size,
                                j * self.stride:j * self.stride + self.kernel_size] * kernel).sum() + bias
        return output

    def conv_mul_bp(self, input: ndarray, kernel: ndarray) -> ndarray:
        output = np.zeros(((input.shape[0] - 1) * self.stride + self.kernel_size,
                           (input.shape[1] - 1) * self.stride + self.kernel_size))

        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                output[i * self.stride:i * self.stride + self.kernel_size,
                j * self.stride:j * self.stride + self.kernel_size] = kernel * input[i, j]

        return output

    def gradient_cal(self, para: ndarray, delta: ndarray) -> ndarray:
        gradient = np.zeros_like(self.gradient_weights)
        para = np.pad(para, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)))

        for para_b, delta_b in zip(para,delta):
            for i in range(self.out_channel):
                for j in range(self.in_channel):
                    for k in range(delta.shape[1]):
                        for m in range(delta.shape[2]):
                            gradient[i, j] += para_b[j, k * self.stride:k * self.stride + self.kernel_size,
                                              m * self.stride:m * self.stride + self.kernel_size] * delta_b[i, k, m]

        return gradient

    @numba.jit(nopython=True)
    def forward(self, input: ndarray) -> ndarray:
        output = np.zeros(
            (input.shape[0], self.out_channel, (input.shape[2] - self.kernel_size + 2 * self.padding) // self.stride + 1
             , (input.shape[3] - self.kernel_size + 2 * self.padding) // self.stride + 1))
        padded_input = np.pad(input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))

        for k in range(input.shape[0]):
            for i in range(self.out_channel):
                for j in range(self.in_channel):
                    output[k, i] += self.conv_mul(padded_input[k, j], self.weights[i, j], self.bias[i])

        return output

    def backward(self, input: ndarray) -> ndarray:
        output = np.zeros((input.shape[0], self.in_channel, (input.shape[2] - 1) * self.stride + self.kernel_size
                           , (input.shape[3] - 1) * self.stride + self.kernel_size))

        for g in range(input.shape[0]):
            for i in range(self.in_channel):
                for j in range(self.out_channel):
                    output[g, i] += self.conv_mul_bp(input[g, j], self.weights[j, i])

        return output[:, :, self.padding:output.shape[1] - self.padding, self.padding:output.shape[2] - self.padding]


class Flatten(Module):

    def __init__(self) -> None:
        self.in_feature = np.zeros(4)

    def forward(self, input: ndarray) -> ndarray:
        self.in_feature = input.shape
        return (input.flatten()).reshape(-1, self.in_feature[0])

    def backward(self, input: ndarray) -> ndarray:
        return input.reshape(self.in_feature)
