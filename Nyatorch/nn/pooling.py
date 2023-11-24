import numpy as np
from numba import cuda
from numpy import ndarray

from .abstract import Module


class MaxPooling(Module):
    def __init__(self, pool_size: int) -> None:
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


class MaxPoolingGPU(Module):
    def __init__(self, pool_size: int) -> None:
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

        output = np.zeros((batch_size, out_height, out_width, channel), dtype=np.float64)

        d_input = cuda.to_device(inputs)
        d_output = cuda.to_device(output)

        block_dim = (out_height, out_width)
        grid_dim = (batch_size, channel)

        _maxpooling_forward_gpu[grid_dim, block_dim](d_input, d_output, self.pool_size)

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

        delta_input = np.zeros((batch_size, in_height, in_width, channel), dtype=np.float64)

        d_delta_output = cuda.to_device(delta_output)
        d_input = cuda.to_device(self.inputs)
        d_delta_input = cuda.to_device(delta_input)

        block_dim = (out_height, out_width)
        grid_dim = (batch_size, channel)

        _maxpooling_backward_gpu[grid_dim, block_dim](d_delta_output, d_input, d_delta_input, self.pool_size)

        d_delta_input.copy_to_host(delta_input)

        del d_delta_output
        del d_input
        del d_delta_input

        return delta_input


@cuda.jit
def _maxpooling_forward_gpu(inputs: ndarray, outputs: ndarray, pool_size: int) -> None:
    """

    :param inputs: [batch_size, in_width, in_height, in_channel]
    :param outputs: [batch_size, out_width, out_height, out_channel]
    :param pool_size: int
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
def _maxpooling_backward_gpu(delta_output: ndarray, inputs: ndarray, delta_input: ndarray, pool_size: int) -> None:
    """

    :param delta_output: [batch_size, out_width, out_height, out_channel]
    :param inputs: [batch_size, in_width, in_height, in_channel]
    :param delta_input: [batch_size, in_width, in_height, in_channel]
    :param pool_size: int
    """
    i, j = cuda.threadIdx.x, cuda.threadIdx.y
    b, c = cuda.blockIdx.x, cuda.blockIdx.y

    if i < delta_output.shape[1] and j < delta_output.shape[2] and c < delta_output.shape[3] \
            and b < delta_output.shape[0]:

        input_height, input_width = inputs.shape[1], inputs.shape[2]

        start_i = i * pool_size
        start_j = j * pool_size
        end_i = min(start_i + pool_size, input_height)
        end_j = min(start_j + pool_size, input_width)

        max_val_index = 0
        max_val = inputs[b, start_i, start_j, c]

        for x in range(start_i, end_i):
            for y in range(start_j, end_j):
                if inputs[b, x, y, c] > max_val:
                    max_val = inputs[b, x, y, c]
                    max_val_index = (x - start_i) * pool_size + (y - start_j)

        max_pos_x = start_i + max_val_index // pool_size
        max_pos_y = start_j + max_val_index % pool_size

        cuda.atomic.add(delta_input, (b, max_pos_x, max_pos_y, c), delta_output[b, i, j, c])
