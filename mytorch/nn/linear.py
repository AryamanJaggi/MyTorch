import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        self.W = np.random.randn(out_features, in_features) * 0.01
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        self.input_shape = A.shape
        A = A.reshape(-1, self.input_shape[-1])
        result = A@self.W.T + self.b
        A = A.reshape(self.input_shape)

        # Store input for backward pass
        self.A = A
        
        result = result.reshape(*self.input_shape[:-1], self.W.shape[0])
        return result

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """

        # Compute gradients
        self.dLdA = dLdZ@self.W

        #want to sum over all but last 2 dimesnions to get dLdW
        dims_to_remove = dLdZ.ndim - 2
        # Create a tuple from 0 up to the number of dims to remove
        axes_to_sum = tuple(range(dims_to_remove))
        #transpose last two dimensions of Z
        dLdZ_swaped = np.swapaxes(dLdZ, -1, -2)
        self.dLdW = np.sum(dLdZ_swaped@self.A, axis = axes_to_sum)

        self.dLdb = np.sum(dLdZ, axis = tuple(range(dLdZ.ndim - 1)))
        
        # Return gradient of loss wrt input
        return self.dLdA
