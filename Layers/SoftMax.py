import numpy as np


'''

- Implementation of SoftMax layer which is used to compute class probabilities

'''


class SoftMax:

    def __init__(self):
        self.y_hat = None

    def forward(self, input_tensor, label_tensor):
        maximums = np.amax(input_tensor, axis=1)

        # For centering the data about the maximum value

        for x in range(input_tensor.shape[0]):
            input_tensor[x, :] = input_tensor[x, :] - maximums[x]

        input_tensor = np.exp(input_tensor)
        sum_input = np.sum(input_tensor, axis=1)

        # Computing the probabilities from the logits

        for x in range(input_tensor.shape[0]):
            input_tensor[x, :] = input_tensor[x, :] / sum_input[x]

        self.y_hat = input_tensor.copy()

        # Computing the Cross Entropy Loss

        loss = 0

        for x in range(label_tensor.shape[0]):
            for y in range(label_tensor.shape[1]):
                if label_tensor[x, y] == 1:
                    loss = loss + (np.log(input_tensor[x, y]) * -1)

        return loss

    '''Same method as above. Added separately to represent the method used during the inference time'''
    def predict(self, input_tensor):
        maximums = np.amax(input_tensor, axis=1)

        for x in range(input_tensor.shape[0]):
            input_tensor[x, :] = input_tensor[x, :] - maximums[x]

        input_tensor = np.exp(input_tensor)
        sum_input = np.sum(input_tensor, axis=1)

        for x in range(input_tensor.shape[0]):
            input_tensor[x, :] = input_tensor[x, :] / sum_input[x]

        return input_tensor

    def backward(self, label_tensor):
        batch_size = label_tensor.shape[0]
        categories = label_tensor.shape[1]
        error_tensor = np.zeros((batch_size, categories), dtype=float)

        # Gradient propagated to the neurons which were fired during the forward pass

        for i in range(batch_size):
            for j in range(categories):
                if label_tensor[i, j] == 1:
                    error_tensor[i, j] = self.y_hat[i, j] - 1
                else:
                    error_tensor[i, j] = self.y_hat[i, j]

        return error_tensor
