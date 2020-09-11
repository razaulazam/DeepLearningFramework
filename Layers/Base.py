from enum import Enum

class BaseClass:
"""Acts a baseclass for the regularization layers. Defines the state of these layers according to the state of the program"""
    def __init__(self):
        self.regularizer = None
        self.flag_set = 0

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer
        self.flag_set = 1


class Phase(Enum):
    train = 1
    test = 2
    validation = 3
