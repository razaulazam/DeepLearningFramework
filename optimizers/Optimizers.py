import numpy as np
import math


class Sgd:
    """Stochastic Gradient Descent"""
    def __init__(self, learning_rate):
        self.global_learning_rate = learning_rate

    def calculate_update(self, individual_delta, weight_tensor, gradient_tensor):
        weight_new_tensor = weight_tensor - (self.global_learning_rate * individual_delta * gradient_tensor)
        return weight_new_tensor


class SgdWithMomentum:
    """Stochastic Gradient Descent with Momentum"""
    def __init__(self, learning_rate, momentum):
        self.global_learning_rate = learning_rate
        self.momentum = momentum
        self.flag = 0
        self.velocity = 0

    def calculate_update(self, individual_delta, weight_tensor, gradient_tensor):
        if self.flag == 0:
            self.velocity = (-1 * self.global_learning_rate * individual_delta) * gradient_tensor
            weight_new_tensor = weight_tensor + self.velocity
            self.flag = 1
        else:
            self.velocity = self.momentum * self.velocity - (self.global_learning_rate * individual_delta *
                                                             gradient_tensor)
            weight_new_tensor = weight_tensor + self.velocity

        return weight_new_tensor


class Adam:
    """Adam optimizer"""
    def __init__(self, learning_rate, momentum, phi):
        self.global_learning_rate = learning_rate
        self.momentum = momentum
        self.phi = phi
        self.flag = 0
        self.exponent = 1
        self.velocity = list()
        self.r_velocity = list()

    def calculate_update(self, individual_delta, weight_tensor, gradient_tensor):
        if self.flag == 0:
            self.velocity.append((1 - self.momentum) * gradient_tensor)
            intermediate_gradient = np.multiply(gradient_tensor, gradient_tensor)
            self.r_velocity.append((1 - self.phi) * intermediate_gradient)
            velocity_hat = self.velocity[0]/(1 - math.pow(self.momentum, self.exponent))
            r_hat = self.r_velocity[0]/(1 - math.pow(self.phi, self.exponent))
            weight_new_tensor = weight_tensor - (individual_delta * self.global_learning_rate *
                                                 (velocity_hat + 0.000000001) / (np.sqrt(np.abs(r_hat)) + 0.000000001))
            self.flag = 1
            self.exponent = self.exponent + 1
        else:
            self.velocity[0] = self.velocity[0] * self.momentum + ((1 - self.momentum) * gradient_tensor)
            intermediate_gradient = np.multiply(gradient_tensor, gradient_tensor)
            self.r_velocity[0] = self.r_velocity[0] * self.phi + ((1 - self.phi) * intermediate_gradient)
            velocity_hat = self.velocity[0] / (1 - math.pow(self.momentum, self.exponent))
            r_hat = self.r_velocity[0] / (1 - math.pow(self.phi, self.exponent))
            weight_new_tensor = weight_tensor - (individual_delta * self.global_learning_rate *
                                                 (velocity_hat + 0.000000001) / (np.sqrt(np.abs(r_hat)) + 0.000000001))
            self.exponent = self.exponent + 1

        return weight_new_tensor
