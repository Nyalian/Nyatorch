import numpy as np
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


class Conv2d(ConvNd):

    def __init__(self, in_channel: int, out_channel: int, kernel_size: int, padding: int = 0, stride: int = 1) -> None:
        """
        Initialize the 2d Convolutional layer.
        The size of the weight is [kernel_size, kernel_size, in_channel, out_channel].
        The size of the bias is [1, out_channel].
        The size of filters is [kernel_size ** 2 * in_channel, out_channel]. (will be updated in forward)

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
        batch_size, in_width, in_height, channel = inputs.shape

        out_width = (in_width - self.pool_size) // self.pool_size + 1
        out_height = (in_height - self.pool_size) // self.pool_size + 1

        max_pool = np.zeros((batch_size, out_width, out_height, channel))

        for y in range(out_width):
            for x in range(out_height):
                x_start, x_end = x * self.pool_size, x * self.pool_size + self.pool_size
                y_start, y_end = y * self.pool_size, y * self.pool_size + self.pool_size

                max_pool[:, y, x, :] = self.inputs[:, y_start:y_end, x_start:x_end, :].max(axis=(1, 2))

        return max_pool

    def backward(self, delta: ndarray) -> ndarray:
        """

        :param delta: [batch_size, out_width, out_height, channel]
        :return: [batch_size, in_width, in_height, channel]
        """

        batch_size, in_width, in_height, channel = self.inputs.shape

        _, out_width, out_height, _ = delta.shape

        d_result = np.zeros((batch_size, in_width, in_height, channel))

        for y in range(out_width):
            for x in range(out_height):
                x_start, x_end = x * self.pool_size, x * self.pool_size + self.pool_size
                y_start, y_end = y * self.pool_size, y * self.pool_size + self.pool_size

                pool = self.inputs[:, y_start:y_end, x_start:x_end, :]
                mask = (pool == np.max(pool, axis=(1, 2), keepdims=True))

                d_result[:, y_start:y_end, x_start:x_end, :] += delta[:, y, x, :].reshape(self.inputs.shape[0], 1, -1,
                                                                                          self.inputs.shape[-1]) * mask

        return d_result


class Flatten(Module):

    def __init__(self) -> None:
        super().__init__()
        self.shape = None

    def forward(self, inputs: ndarray) -> ndarray:
        self.shape = inputs.shape
        batch_size, width, height, channel = inputs.shape
        return (inputs.flatten()).reshape(batch_size, width * height * channel)

    def backward(self, delta: ndarray) -> ndarray:
        return delta.reshape(self.shape)
