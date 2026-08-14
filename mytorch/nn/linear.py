import numpy as np


class Linear:
    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros
        """
        self.debug = debug
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros((out_features,))  # when we broadcast it shape will become (N,outfeatures)

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output of linear layer with shape (N, C1)

        """
        self.A = A
        self.N = A.shape[0]  # store the batch size parameter of the input A

        Z = A@self.W.T + self.b  # numpy native broadcasting will extend b and make the shape (N,outfeatures)
        return Z 

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (N, C1)
        :return: Gradient of loss wrt input A (N, C0)

        """
        dLdA = dLdZ @ self.W
        self.dLdW = dLdZ.T @ self.A
        self.dLdb = np.sum(dLdZ, axis=0)

        if self.debug:
            self.dLdA = dLdA

        return dLdA
