from typing import Dict

import numpy as np
from numba import cuda
from numpy import ndarray

from .abstract import ConvNd


class Conv2d(ConvNd):

    def __init__(self, in_channel: int, out_channel: int, kernel_size: int, padding: int = 0, stride: int = 1) -> None:
        """
        Initialize the Conv2d layer.
        The size of the weight is [kernel_size, kernel_size, in_channel, out_channel].
        The size of the bias is [1, out_channel].

        :param in_channel: The input channel
        :param out_channel: The output channel
        :param kernel_size: The kernel size
        :param padding: The padding size
        :param stride: The stride
        """
        super().__init__(in_channel, out_channel, kernel_size, padding, stride)
        bound = np.sqrt(6. / (in_channel * kernel_size ** 2 + out_channel * kernel_size ** 2))
        self.weights = np.random.uniform(-bound, bound, (kernel_size, kernel_size, self.in_channel, self.out_channel))
        self.bias = np.random.rand(1, self.out_channel)
        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias = np.zeros_like(self.bias)
        self.filters = None
        self.inputs = None

    def forward(self, inputs: ndarray) -> ndarray:
        """
        Forward calculation of the Conv2d layer.

        :param inputs: [batch_size, in_width, in_height, in_channel]
        :return: [batch_size, out_width, out_height, out_channel]
        """
        self.inputs = inputs
        batch_size, in_width, in_height, in_channel = inputs.shape

        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1

        output = np.zeros((batch_size, out_width, out_height, self.out_channel))

        padded_input = np.pad(inputs, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))

        # [kernel_size1, kernel_size2, in_channel, out_channel] to
        # [in_channel * kernel_size2 * kernel_size1 , out_channel]
        self.filters = self.weights.reshape(self.kernel_size ** 2 * in_channel, self.out_channel)

        for y in range(out_width):
            for x in range(out_height):
                y_start, y_end = y * self.stride, y * self.stride + self.kernel_size
                x_start, x_end = x * self.stride, x * self.stride + self.kernel_size
                receptive_area = padded_input[:, y_start:y_end, x_start:x_end, :]
                receptive_area = receptive_area.reshape(batch_size, self.kernel_size ** 2 * in_channel)
                # [batch_size, kernel_size1, kernel_size2, in_channel] to
                # [batch_size, in_channel * kernel_size2 * kernel_size1]

                output[:, y, x, :] = receptive_area @ self.filters + self.bias
        return output

    def backward(self, delta: ndarray) -> ndarray:
        """
        Backpropagation of the Conv2d layer.

        :param delta: [batch_size, out_width, out_height, out_channel]
        :return: [batch_size, in_width, in_height, in_channel]
        """

        batch_size, in_width, in_height, in_channel = self.inputs.shape

        _, out_width, out_height, out_channel = delta.shape

        d_result = np.zeros((batch_size, in_width, in_height, in_channel))

        for y in range(out_width):
            for x in range(out_height):
                y_start, y_end = y * self.stride, y * self.stride + self.kernel_size
                x_start, x_end = x * self.stride, x * self.stride + self.kernel_size
                # [batch_size, out_channel] @ [out_channel, in_channel * kernel_size2 * kernel_size1]
                d_wrt_input = delta[:, y, x, :] @ self.filters.T

                # [batch_size, in_channel, kernel_size2, kernel_size1]
                d_wrt_input = d_wrt_input.reshape(batch_size, in_channel, self.kernel_size, self.kernel_size)
                # [batch_size, kernel_size1, kernel_size2, in_channel]
                d_wrt_input = d_wrt_input.swapaxes(1, 3)

                d_result[:, y_start:y_end, x_start:x_end, :] += d_wrt_input
                # [batch_size, kernel_size1, kernel_size2, in_channel].T
                # [in_channel, kernel_size2, kernel_size1, batch_size] @ [batch_size, out_channel]
                # [in_channel, kernel_size2, kernel_size1, out_channel]
                tmp = self.inputs[:, y_start:y_end, x_start:x_end, :].T @ delta[:, y, x, :]

                tmp = tmp.swapaxes(0, 2)
                # [kernel_size1, kernel_size2, in_channel, out_channel]

                self.gradient_weights += tmp
                self.gradient_bias += np.sum(delta[:, y, x, :], axis=0)

        return d_result

    def get_parameter(self) -> Dict[str, ndarray]:
        return {"weights": self.weights, "bias": self.bias}

    def set_parameter(self, parameter: Dict[str, ndarray]) -> None:
        self.weights = parameter["weights"]
        self.bias = parameter["bias"]

    def update(self, learning_rate: int) -> None:
        self.weights -= learning_rate * self.gradient_weights
        self.bias -= learning_rate * self.gradient_bias


class Conv2dGPU(ConvNd):

    def __init__(self, in_channel: int, out_channel: int, kernel_size: int, padding: int = 0, stride: int = 1) -> None:
        """
        Initialize the 2d Convolutional layer.
        The size of the weight is [kernel_size, kernel_size, in_channel, out_channel].
        The size of the bias is [1, out_channel].

        :param in_channel: The input channel
        :param out_channel: The output channel
        :param kernel_size: The kernel size
        :param padding: The padding size
        :param stride: The stride
        """
        super().__init__(in_channel, out_channel, kernel_size, padding, stride)
        bound = np.sqrt(2. / (in_channel * kernel_size ** 2 + out_channel * kernel_size ** 2))
        self.weights = np.random.uniform(-bound, bound, (kernel_size, kernel_size, self.in_channel, self.out_channel))
        self.bias = np.random.rand(1, self.out_channel)
        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias = np.zeros_like(self.bias)
        self.inputs = None

    def forward(self, inputs: ndarray) -> ndarray:
        """
        Forward calculation of the Conv2d layer.

        :param inputs: [batch_size, in_width, in_height, in_channel]
        :return: [batch_size, out_width, out_height, out_channel]
        """

        batch_size, in_width, in_height, in_channel = inputs.shape

        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1

        output = np.zeros((batch_size, out_width, out_height, self.out_channel))

        padded_input = np.pad(inputs, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))

        self.inputs = inputs

        d_input = cuda.to_device(padded_input)
        d_kernel = cuda.to_device(self.weights)
        d_output = cuda.to_device(output)

        block_dim = (out_width, out_height)
        grid_dim = (batch_size, self.out_channel)

        _convolution_gpu[grid_dim, block_dim](d_input, d_kernel, d_output)

        d_output.copy_to_host(output)

        del d_input
        del d_kernel
        del d_output

        return output + self.bias

    def backward(self, delta_output: ndarray) -> ndarray:
        """
        Backpropagation of the Conv2d layer.

        :param delta_output: [batch_size, out_width, out_height, out_channel]
        :return: [batch_size, in_width, in_height, in_channel]
        """

        batch_size, in_width, in_height, in_channel = self.inputs.shape

        _, out_width, out_height, out_channel = delta_output.shape

        delta_input = np.zeros((batch_size, in_width, in_height, in_channel), dtype=np.float64)
        gradient_weights = np.zeros_like(self.gradient_weights, dtype=np.float64)

        d_delta_output = cuda.to_device(delta_output)
        d_input = cuda.to_device(self.inputs)
        d_delta_input = cuda.to_device(delta_input)
        d_weight = cuda.to_device(self.weights)
        d_gradient_weights = cuda.to_device(gradient_weights)

        block_dim1 = (in_width, in_height)
        grid_dim1 = (batch_size, in_channel)
        block_dim2 = (self.kernel_size, self.kernel_size)
        grid_dim2 = (in_channel, out_channel)

        _convolution_backward_gpu[grid_dim1, block_dim1](d_delta_output, d_weight, d_delta_input)
        _convolution_kernel_gpu[grid_dim2, block_dim2](d_delta_output, d_input, d_gradient_weights)

        d_delta_input.copy_to_host(delta_input)
        d_gradient_weights.copy_to_host(gradient_weights)

        self.gradient_weights = gradient_weights
        self.gradient_bias = delta_output.sum(axis=(0, 1, 2)).reshape(1, -1)

        del d_delta_output
        del d_input
        del d_delta_input
        del d_weight
        del d_gradient_weights

        return delta_input

    def get_parameter(self) -> Dict[str, ndarray]:
        return {"weights": self.weights, "bias": self.bias}

    def set_parameter(self, parameter: Dict[str, ndarray]) -> None:
        self.weights = parameter["weights"]
        self.bias = parameter["bias"]

    def update(self, learning_rate: int) -> None:
        self.weights -= learning_rate * self.gradient_weights
        self.bias -= learning_rate * self.gradient_bias


@cuda.jit
def _convolution_gpu(inputs: ndarray, weights: ndarray, outputs: ndarray) -> None:
    """

    :param inputs: [batch_size, in_width, in_height, in_channel]
    :param weights: [kernel_size, kernel_size, in_channel, out_channel]
    :param outputs: [batch_size, out_width, out_height, out_channel]
    """
    i, j = cuda.threadIdx.x, cuda.threadIdx.y
    b, out_c = cuda.blockIdx.x, cuda.blockIdx.y

    kernel_size, _, _, _ = weights.shape

    value = 0.0
    for x in range(kernel_size):
        for y in range(kernel_size):
            for in_c in range(inputs.shape[3]):
                value += weights[x, y, in_c, out_c] * inputs[b, i + x, j + y, in_c]

        outputs[b, i, j, out_c] = value


@cuda.jit
def _convolution_backward_gpu(delta_output: ndarray, weights: ndarray, delta_input: ndarray) -> None:
    """

    :param delta_output: [batch_size, out_width, out_height, out_channel]
    :param weights: [kernel_size, kernel_size, in_channel, out_channel]
    :param delta_input: [batch_size, in_width, in_height, in_channel]
    """
    in_w, in_h = cuda.threadIdx.x, cuda.threadIdx.y
    b_size, in_c = cuda.blockIdx.x, cuda.blockIdx.y

    kernel_size, _, _, out_c = weights.shape
    _, out_w, out_h, _ = delta_output.shape

    value = 0.0
    for m in range(out_c):
        for k in range(kernel_size):
            for l in range(kernel_size):
                if 0 <= in_w - k < out_w and 0 <= in_h - l < out_h:
                    value += delta_output[b_size, in_w - k, in_h - l, m] * weights[k, l, in_c, m]

    delta_input[b_size, in_w, in_h, in_c] = value


@cuda.jit
def _convolution_kernel_gpu(delta_output: ndarray, inputs: ndarray, gradient_weights: ndarray) -> None:
    """

    :param delta_output: [batch_size, out_width, out_height, out_channel]
    :param inputs: [batch_size, in_width, in_height, in_channel]
    :param gradient_weights: [kernel_size, kernel_size, in_channel, out_channel]
    """
    # in_width > out_width
    # k, l ~ [0, kernel_size)
    # if stride = 1, padding = 1
    # in_width - kernel_size + 1 = out_width
    # kernel_size - 1 + out_width - 1 = in_width - 1
    # i + k ~ [0, in_width)
    kernel_w, kernel_h = cuda.threadIdx.x, cuda.threadIdx.y
    in_c, out_c = cuda.blockIdx.x, cuda.blockIdx.y

    batch_size, out_width, out_height, _ = delta_output.shape
    value = 0.0
    for b in range(batch_size):
        for i in range(out_width):
            for j in range(out_height):
                value += delta_output[b, i, j, out_c] * inputs[b, i + kernel_w, j + kernel_h, in_c]

    gradient_weights[kernel_w, kernel_h, in_c, out_c] = value
