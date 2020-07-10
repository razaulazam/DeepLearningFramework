import numpy as np


'''

- Implementation of fully connected layers which are commonly used in Feed Forward Neural Networks
- Forward method is called during the forward pass at train/test time
- Backward method is called during the backward pass at train time

'''

class FullyConnected:

    def __init__(self, input_size, output_size):
        '''
        Args:
        
        - input_size (int)
        - output_size (int)
        
        
        Attributes:
        
        - input_size: number of neurons in the previous layer
        - output_size: number of neurons in the current layer which we are building
        '''
        total_size = input_size + 1
        self.weights = np.random.random([total_size, output_size]) # Uniform random weight initialization
        self.delta = 1 # Initial value of the learning rate which would be optimized during training
        self.input = None
        self.gradient = None

    def forward(self, input_tensor):
        '''

        param: input_tensor - input to the fully connected layer
        
        returns: output_tensor - output from the layer during the forward pass computation for the next layer

        '''
        batch_size = input_tensor.shape[0]
        bias = np.ones((batch_size, 1), dtype=int)

        input_tensor = np.hstack((input_tensor, bias))

        self.input = input_tensor.copy()
        input_weight_transpose = np.transpose(self.weights)
        input_tensor_transpose = np.transpose(input_tensor)

        # output = (w * input) + bias

        output_tensor = np.dot(input_weight_transpose, input_tensor_transpose)

        return np.transpose(output_tensor)

    def backward(self, error_tensor):
        '''

        param: error_tensor - error signal received by the layer during the backward pass
        
        returns: new_error_tensor - error signal for the previous layer in the network during the backward pass

        '''
        input_weight_transpose = np.transpose(self.weights)

        new_tensor = np.dot(error_tensor, input_weight_transpose)
        num_cols = new_tensor.shape[1] - 1
        new_error_tensor = np.delete(new_tensor, num_cols, 1)
        input_transpose = np.transpose(self.input)

        self.gradient = np.dot(input_transpose, error_tensor)
        self.weights -= self.delta*self.gradient # performing weight updates

        return new_error_tensor

    def get_gradient_weights(self):
         '''Returns the gradient which can be used to verify the computations using e.g. numerical gradient computation'''
        return self.gradient
