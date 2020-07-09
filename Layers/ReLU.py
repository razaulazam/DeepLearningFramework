import numpy as np

'''

- Implementation of ReLU non-linearity which is commonly used in almost all Deep Neural Networks
- Helps with combatting vanishing gradient problems with a careful selection of learning rate


'''


class ReLU:

    def __init__(self):
        self.input = 0

    def forward(self, input_tensor):
        '''

        param: input_tensor - input to the ReLU layer

        '''
        self.input = input_tensor.copy()
        output_tensor = np.maximum(0, input_tensor)
        return output_tensor

    def backward(self, error_tensor):
        '''

        param: error_tensor - error signal received by the layer during the backward pass

        '''
        batch_size = self.input.shape[0]
        input_size = self.input.shape[1]
        new_error_tensor = np.zeros((batch_size, input_size), dtype=float)
        for i in range(batch_size):
            for j in range(input_size):
                if self.input[i, j] > 0:
                    new_error_tensor[i, j] = error_tensor[i, j]

        return new_error_tensor
