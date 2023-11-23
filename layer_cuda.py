import numpy as np
from numba import cuda
from numpy import ndarray

from module import Module


class LinearLayer(Module):

    def __init__(self, in_features: int, out_features: int) -> None:
        """
        Initialize the fully connected layer.
        The size of the weight is [in_features, out_features].
        The size of the bias is [1, out_features].

        :param in_features: The input features
        :param out_features: The output features
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        bound = np.sqrt(6. / (self.in_features + self.out_features))
        self.weights = np.random.uniform(-bound, bound, (self.in_features, self.out_features))
        self.bias = np.random.rand(1, out_features)
        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias = np.zeros_like(self.bias)
        self.inputs = None

    def forward(self, inputs: ndarray) -> ndarray:
        """
        Forward calculation of the fully connected layer.

        :param inputs: [batch_size, in_features]
        :return: [batch_size, out_features]
        """
        self.inputs = inputs
        return inputs @ self.weights + self.bias

    def backward(self, delta: ndarray) -> ndarray:
        """
        Backpropagation of the fully connected layer.

        :param delta: [batch_size, out_features]
        :return: [batch_size, in_features]
        """
        self.gradient_weights = self.inputs.T @ delta
        self.gradient_bias = delta.sum(axis=0, keepdims=True)
        return delta @ self.weights.T

    def get_parameter(self):
        return {"weights": self.weights, "bias": self.bias}

    def set_parameter(self, parameter: dict):
        self.weights = parameter["weights"]
        self.bias = parameter["bias"]

    def update(self, learning_rate):
        self.weights -= learning_rate * self.gradient_weights
        self.bias -= learning_rate * self.gradient_bias


class ConvNd(Module):

    def __init__(self, in_channel: int, out_channel: int, kernel_size: int, padding: int, stride: int) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.weights = None
        self.bias = None
        self.gradient_weights = None
        self.gradient_bias = None

    def forward(self, inputs: ndarray) -> ndarray:
        pass

    def backward(self, delta: ndarray) -> ndarray:
        pass

    def update(self, learning_rate):
        self.weights -= learning_rate * self.gradient_weights
        self.bias -= learning_rate * self.gradient_bias


class Conv2d(ConvNd):

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
        bound = np.sqrt(6. / (in_channel * kernel_size ** 2 + out_channel * kernel_size ** 2))
        self.weights = np.random.uniform(-bound, bound, (kernel_size, kernel_size, self.in_channel, self.out_channel))
        # self.weights = np.random.randn(kernel_size, kernel_size, self.in_channel, self.out_channel)
        self.bias = np.random.rand(1, self.out_channel)
        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias = np.zeros_like(self.bias)
        self.filters = None
        self.inputs = None

    def forward(self, inputs: ndarray) -> ndarray:
        """

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

        convolution_gpu[grid_dim, block_dim](d_input, d_kernel, d_output)

        d_output.copy_to_host(output)

        del d_input
        del d_kernel
        del d_output

        return output + self.bias

    def backward(self, delta_output: ndarray) -> ndarray:
        """
        The size of the weight is [kernel_size, kernel_size, in_channel, out_channel].
        The size of the bias is [1, out_channel].

        :param delta_output: [batch_size, out_width, out_height, out_channel]
        :return: [batch_size, in_width, in_height, in_channel]
        """

        batch_size, in_width, in_height, in_channel = self.inputs.shape

        _, out_width, out_height, out_channel = delta_output.shape

        delta_input = np.zeros((batch_size, in_width, in_height, in_channel), dtype=np.float32)
        gradient_weights = np.zeros_like(self.gradient_weights, dtype=np.float32)

        d_delta_output = cuda.to_device(delta_output)
        d_input = cuda.to_device(self.inputs)
        d_delta_input = cuda.to_device(delta_input)
        d_weight = cuda.to_device(self.weights)
        d_gradient_weights = cuda.to_device(gradient_weights)

        block_dim1 = (in_width, in_height)
        grid_dim1 = (batch_size, in_channel)
        block_dim2 = (self.kernel_size, self.kernel_size)
        grid_dim2 = (in_channel, out_channel)

        convolution_backward_gpu[grid_dim1, block_dim1](d_delta_output, d_weight, d_delta_input)
        convolution_kernel_gpu[grid_dim2, block_dim2](d_delta_output, d_input, d_gradient_weights)

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

    def get_parameter(self):
        return {"weights": self.weights, "bias": self.bias}

    def set_parameter(self, parameter: dict):
        self.weights = parameter["weights"]
        self.bias = parameter["bias"]

    def update(self, learning_rate):
        self.weights -= learning_rate * self.gradient_weights
        self.bias -= learning_rate * self.gradient_bias


class MaxPooling(Module):
    def __init__(self, pool_size: int):
        super().__init__()
        self.pool_size = pool_size
        self.inputs = None

    def forward(self, inputs: ndarray) -> ndarray:
        """

        :param inputs: [batch_size, in_width, in_height, channel]
        :return: [batch_size, out_width, out_height, channel]
        """
        self.inputs = inputs
        batch_size, in_height, in_width, channel = inputs.shape

        out_height = (in_height - self.pool_size) // self.pool_size + 1
        out_width = (in_width - self.pool_size) // self.pool_size + 1

        output = np.zeros((batch_size, out_height, out_width, channel), dtype=np.float32)

        d_input = cuda.to_device(inputs)
        d_output = cuda.to_device(output)

        block_dim = (out_height, out_width)
        grid_dim = (batch_size, channel)

        maxpooling_forward_gpu[grid_dim, block_dim](d_input, d_output, self.pool_size)

        d_output.copy_to_host(output)

        del d_input
        del d_output

        return output

    def backward(self, delta_output: ndarray) -> ndarray:
        """

        :param delta_output: [batch_size, out_width, out_height, channel]
        :return: [batch_size, in_width, in_height, channel]
        """

        batch_size, in_height, in_width, channel = self.inputs.shape

        _, out_height, out_width, _ = delta_output.shape

        delta_input = np.zeros((batch_size, in_height, in_width, channel), dtype=np.float32)

        d_delta_output = cuda.to_device(delta_output)
        d_input = cuda.to_device(self.inputs)
        d_delta_input = cuda.to_device(delta_input)

        block_dim = (out_height, out_width)
        grid_dim = (batch_size, channel)

        maxpooling_backward_gpu[grid_dim, block_dim](d_delta_output, d_input, d_delta_input, self.pool_size)

        d_delta_input.copy_to_host(delta_input)

        del d_delta_output
        del d_input
        del d_delta_input

        return delta_input


class Flatten(Module):

    def __init__(self) -> None:
        super().__init__()
        self.shape = None

    def forward(self, inputs: ndarray) -> ndarray:
        self.shape = inputs.shape
        batch_size, height, width, channel = inputs.shape
        return (inputs.flatten()).reshape(batch_size, height * width * channel)

    def backward(self, delta: ndarray) -> ndarray:
        return delta.reshape(self.shape)


@cuda.jit
def convolution_gpu(inputs, weights, outputs):
    i, j = cuda.threadIdx.x, cuda.threadIdx.y
    b, out_c = cuda.blockIdx.x, cuda.blockIdx.y

    kw, kh, _, _ = weights.shape

    value = 0.0
    for x in range(kh):
        for y in range(kw):
            for in_c in range(inputs.shape[3]):
                value += weights[x, y, in_c, out_c] * inputs[b, i + x, j + y, in_c]

        outputs[b, i, j, out_c] = value


@cuda.jit
def convolution_backward_gpu(delta_output, weight, delta_input):
    """
    :param delta_output: [batch_size, out_width, out_height, out_channel]
    :param weight: [kernel_size, kernel_size, in_channel, out_channel]
    :param delta_input: [batch_size, in_width, in_height, in_channel]
    """
    in_w, in_h = cuda.threadIdx.x, cuda.threadIdx.y
    b_size, in_c = cuda.blockIdx.x, cuda.blockIdx.y

    kernel_size, _, _, out_c = weight.shape
    _, out_w, out_h, _ = delta_output.shape

    value = 0.0
    for m in range(out_c):
        for k in range(kernel_size):
            for l in range(kernel_size):
                if 0 <= in_w - k < out_w and 0 <= in_h - l < out_h:
                    value += delta_output[b_size, in_w - k, in_h - l, m] * weight[k, l, in_c, m]

    delta_input[b_size, in_w, in_h, in_c] = value


@cuda.jit
def convolution_kernel_gpu(delta_output, input, gradient_weights):
    """
    in_width > out_width
    k, l ~ [0, kernel_size)
    if stride = 1, padding = 1
    in_width - kernel_size + 1 = out_width
    kernel_size - 1 + out_width - 1 = in_width - 1
    i + k ~ [0, in_width)
    :param delta_output: [batch_size, out_width, out_height, out_channel]
    :param input: [batch_size, in_width, in_height, in_channel]
    :param gradient_weights: [kernel_size, kernel_size, in_channel, out_channel]
    """
    kernel_w, kernel_h = cuda.threadIdx.x, cuda.threadIdx.y
    in_c, out_c = cuda.blockIdx.x, cuda.blockIdx.y

    batch_size, out_width, out_height, _ = delta_output.shape
    value = 0.0
    for b in range(batch_size):
        for i in range(out_width):
            for j in range(out_height):
                value += delta_output[b, i, j, out_c] * input[b, i + kernel_w, j + kernel_h, in_c]

    gradient_weights[kernel_w, kernel_h, in_c, out_c] = value


@cuda.jit
def maxpooling_forward_gpu(inputs, outputs, pool_size):
    """

    :param inputs:
    :param outputs:
    :param pool_size:
    :return:
    """
    i, j = cuda.threadIdx.x, cuda.threadIdx.y
    b, c = cuda.blockIdx.x, cuda.blockIdx.y

    if i < outputs.shape[1] and j < outputs.shape[2] and c < outputs.shape[3] and b < inputs.shape[0]:
        start_i = i * pool_size
        start_j = j * pool_size
        end_i = min(start_i + pool_size, inputs.shape[1])
        end_j = min(start_j + pool_size, inputs.shape[2])

        max_val = inputs[b, start_i, start_j, c]

        for x in range(start_i, end_i):
            for y in range(start_j, end_j):
                max_val = max(max_val, inputs[b, x, y, c])

        outputs[b, i, j, c] = max_val


@cuda.jit
def maxpooling_backward_gpu(delta_output, inputs, delta_input, pool_size):
    i, j = cuda.threadIdx.x, cuda.threadIdx.y
    b, c = cuda.blockIdx.x, cuda.blockIdx.y

    if i < delta_output.shape[1] and j < delta_output.shape[2] and c < delta_output.shape[3] \
            and b < delta_output.shape[0]:

        input_height, input_width = inputs.shape[1], inputs.shape[2]

        start_i = i * pool_size
        start_j = j * pool_size
        end_i = min(start_i + pool_size, input_height)
        end_j = min(start_j + pool_size, input_width)

        # Find the position of the max value in the corresponding pooling window
        max_val_index = 0
        max_val = inputs[b, start_i, start_j, c]

        for x in range(start_i, end_i):
            for y in range(start_j, end_j):
                if inputs[b, x, y, c] > max_val:
                    max_val = inputs[b, x, y, c]
                    max_val_index = (x - start_i) * pool_size + (y - start_j)

        # Compute the position in the input array corresponding to the max value
        max_pos_x = start_i + max_val_index // pool_size
        max_pos_y = start_j + max_val_index % pool_size

        # Accumulate the gradient at the position of the max value
        cuda.atomic.add(delta_input, (b, max_pos_x, max_pos_y, c), delta_output[b, i, j, c])
