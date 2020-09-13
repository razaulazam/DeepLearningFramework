import numpy as np

class TanH:
    """Implementation of the TanH non-linearity"""
    def __init__(self):
        self.activations = None

    def forward(self, input_tensor):
        self.activations = np.tanh(input_tensor)
        return self.activations

    def backward(self, error_tensor):
        der = 1 - self.activations**2
        return error_tensor * der
