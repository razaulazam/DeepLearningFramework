import numpy as np
import math


class Constant:
    """Constant initialization"""
    def __init__(self, val):
        self.value = val
        self.weights = []
        self.fan_in = 0
        self.fan_out = 0

    def initialize(self, weights_shape, fan_in, fan_out):
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.weights = np.zeros(weights_shape, dtype=float)
        self.weights[:, :] = self.value
        return self.weights


class UniformRandom:
    """Uniform random initialization"""
    def __init__(self):
        self.weights = []
        self.fan_in = 0
        self.fan_out = 0

    def initialize(self, weights_shape, fan_in, fan_out):
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.weights = np.random.random(weights_shape)
        return self.weights


class Xavier:
    """Xavier initialization"""
    def __init__(self):
        self.mean = 0
        self.sigma = 0
        self.fan_in = 0
        self.fan_out = 0
        self.weights = []

    def initialize(self, weights_shape, fan_in, fan_out):
        self.fan_in = fan_in
        self.fan_out = fan_out
        num = 2/(fan_out+fan_in)
        self.sigma = math.sqrt(num)
        self.weights = np.random.normal(self.mean, self.sigma, weights_shape)
        return self.weights


class He:
    """He initialization"""
    def __init__(self):
        self.mean = 0
        self.sigma = 0
        self.fan_in = 0
        self.fan_out = 0
        self.weights = []

    def initialize(self, weights_shape, fan_in, fan_out):
        self.fan_in = fan_in
        self.fan_out = fan_out
        num = 2/fan_in
        self.sigma = math.sqrt(num)
        self.weights = np.random.normal(self.mean, self.sigma, weights_shape)
        return self.weights
