import numpy as np
from Layers import Base


class Dropout(Base.BaseClass):
    """Implementation of the Dropout layer"""
    def __init__(self, probability):
        """
        Attributes:
        - probability: dropout probability (parameter required for the Bernouli distribution)
        """
        self.probability = probability
        self.phase = Base.Phase.train
        self.drop = 0
        self.new_input_tensor = 0
        super().__init__()

    def forward(self, input_tensor):
        """Does the forward pass computations"""
        if self.phase == Base.Phase.train:
            self.drop = np.random.binomial(1, self.probability, size=input_tensor.shape)
            new_input = np.multiply(input_tensor, self.drop)
            new_input /= self.probability
            self.new_input_tensor = new_input
        else:
            self.new_input_tensor = input_tensor

        return self.new_input_tensor

    def backward(self, error_tensor):
        """Does the backward pass computations"""
        error = np.multiply(self.drop, error_tensor)
        return error
