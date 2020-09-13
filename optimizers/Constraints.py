import numpy as np

class L2_Regularizer:
    """L2 regularization"""
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate(self, weights):
        return self.alpha * weights

    def norm(self, weights):
        return self.alpha * np.sqrt(np.sum(np.power(weights, 2)))


class L1_Regularizer:
    """L1 regularization"""
    def __init__(self,alpha):
        self.alpha = alpha

    def calculate(self, weights):
        return self.alpha * np.sign(weights)

    def norm(self, weights):
        return self.alpha * np.sum(np.abs(weights))
