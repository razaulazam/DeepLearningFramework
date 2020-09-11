import numpy as np
import math


class Pooling:

    def __init__(self, stride_shape, pooling_shape):
        """
        Attributes:
        - stride_shape: defines the stride values that are used in the height and the width dimension of an image
        - pooling_shape: shape of the kernel (window)
        """
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.indices = 0
        self.input = 0

    def forward(self, input_tensor):
        """Performs the forward pass computations"""
        batch_size = input_tensor.shape[0]
        channels = input_tensor.shape[1]
        input_tensor_height = input_tensor.shape[2]
        input_tensor_width = input_tensor.shape[3]
        dim_row = math.floor((input_tensor_height - self.pooling_shape[0]) / self.stride_shape[0]) + 1
        dim_col = math.floor((input_tensor_width - self.pooling_shape[1]) / self.stride_shape[1]) + 1

        pool = np.zeros((batch_size, channels, dim_row, dim_col))
        self.input = np.zeros(input_tensor.shape)
        self.indices = list()
        for batch in range(batch_size):
            for chan in range(channels):
                r2 = 0
                for r in np.arange(0, input_tensor_height - self.pooling_shape[0] + 1, self.stride_shape[0]):
                    c2 = 0
                    for c in np.arange(0, input_tensor_width - self.pooling_shape[1] + 1, self.stride_shape[1]):
                        temp = input_tensor[batch, chan, r:r + self.pooling_shape[0], c:c + self.pooling_shape[1]]
                        index = np.unravel_index(np.argmax(temp, axis=None), temp.shape)
                        maximum = np.max([temp])
                        self.indices.append((r+index[0], c+index[1]))
                        self.input[batch, chan, r+index[0], c+index[1]] += 1
                        pool[batch, chan, r2, c2] = maximum
                        c2 = c2 + 1
                    r2 = r2 + 1
        return pool

    def backward(self, error_tensor):
        """Performs the backward pass computations"""
        batch_size = error_tensor.shape[0]
        channels = error_tensor.shape[1]

        visits = np.zeros(self.input.shape)
        count = 0
        for batch in range(batch_size):
            for chan in range(channels):
                for row in range(error_tensor.shape[2]):
                    for col in range(error_tensor.shape[3]):
                        index = self.indices[count]
                        count = count + 1

                        if visits[batch,chan, index[0], index[1]] == 1:
                            self.input[batch,chan, index[0], index[1]] += error_tensor[batch, chan, row, col]
                        else:
                            self.input[batch, chan, index[0], index[1]] = error_tensor[batch, chan, row, col]

                        visits[batch, chan, index[0], index[1]] = 1

        return self.input
