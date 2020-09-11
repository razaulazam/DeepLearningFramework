import numpy as np


class Flatten:
    """Flattens the tensor"""
    def __init__(self):
        self.batch_size = 0
        self.dim1 = 0
        self.dim2 = 0
        self.dim3 = 0

    def forward(self, input_tensor):
        """Does the forward pass computations"""
        (self.batch_size, self.dim1, self.dim2, self.dim3) = input_tensor.shape
        input_tensor = input_tensor.reshape(self.batch_size, np.prod(self.dim1 * self.dim2 * self.dim3))
        return input_tensor

    def backward(self, error_tensor):
        """Does the backward pass computations"""
        error_tensor = error_tensor.reshape(self.batch_size, self.dim1, self.dim2, self.dim3)
        return error_tensor
