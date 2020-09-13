import numpy as np


class Sigmoid:
    """Implementation of the sigmoid non-linearity"""
    def __init__(self):
        self.activations = None

    def forward(self, input_tensor):
        self.activations = 1/(1 + np.exp(-input_tensor))
        return self.activations

    def backward(self, error_tensor):
        der = self.activations * (1 - self.activations)
        return error_tensor * der
