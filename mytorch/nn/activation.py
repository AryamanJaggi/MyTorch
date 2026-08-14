import numpy as np
import scipy


### No need to modify Identity class
class Identity:
    """
    Identity activation function.
    """

    def forward(self, Z):
        """
        :param Z: Batch of data Z (N samples, C features) to apply activation function to input Z.
        :return: Output returns the computed output A (N samples, C features).
        """
        self.A = Z
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt post-activation output (a measure of how the output A affect the loss L)
        :return: Gradient of loss with respect to pre-activation input (a measure of how the input Z affect the loss L)
        """
        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ
        return dLdZ


class Sigmoid:
    """
    Sigmoid activation function.
    """
    def forward(self, Z):
        """
        :param Z: Batch of data Z (N samples, C features) to apply activation function to input Z.
        :return: Output returns the computed output A (N samples, C features).
        """
        self.A = 1/(1+np.exp(-Z))
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt post-activation output (a measure of how the output A affect the loss L)
        :return: Gradient of loss with respect to pre-activation input (a measure of how the input Z affect the loss L)
        """
        dAdZ = (self.A-(self.A*self.A))
        dLdZ = dLdA * dAdZ
        return dLdZ


class Tanh:
    """
    Tanh activation function.
    """
    def forward(self, Z):
        self.A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
        return self.A

    def backward(self, dLdA):
        dAdZ = 1 - self.A * self.A
        return dLdA * dAdZ


class ReLU:
    """
    ReLU (Rectified Linear Unit) activation function.
    """
    def forward(self, Z):
        self.A = np.maximum(0, Z)
        return self.A

    def backward(self, dLdA):
        dAdZ = np.where(self.A > 0, 1, 0)
        return dLdA * dAdZ


class GELU:
    """
    GELU (Gaussian Error Linear Unit) activation function.

    Forward:  A = 0.5 * Z * (1 + erf(Z / sqrt(2)))
    Backward: dA/dZ = 0.5 * (1 + erf(Z / sqrt(2))) + Z * (1/sqrt(2*pi)) * exp(-Z^2 / 2)
    """
    def forward(self, Z):
        self.Z = Z
        self.A = 0.5 * Z * (1 + scipy.special.erf(Z / np.sqrt(2)))
        return self.A

    def backward(self, dLdA):
        dAdZ = 0.5 * (1 + scipy.special.erf(self.Z / np.sqrt(2))) + \
               self.Z * (1 / np.sqrt(2 * np.pi)) * np.exp(-(self.Z ** 2) / 2)
        dLdZ = dLdA * dAdZ
        return dLdZ


class Swish:
    """
    Swish activation function: A = Z * sigmoid(beta * Z), with a learnable scalar beta.

    Forward:  A = Z * sigmoid(beta * Z)
    Backward: dA/dZ    = sigmoid(beta*Z) + beta*Z*sigmoid(beta*Z)*(1 - sigmoid(beta*Z))
              dL/dbeta = sum( dLdA * Z^2 * sigmoid(beta*Z) * (1 - sigmoid(beta*Z)) )
    """
    def __init__(self, beta=1.0):
        self.beta = beta

    def forward(self, Z):
        self.Z = Z
        self.sig = 1 / (1 + np.exp(-self.beta * Z))
        self.A = Z * self.sig
        return self.A

    def backward(self, dLdA):
        dAdZ = self.sig + self.beta * self.Z * self.sig * (1 - self.sig)
        dLdZ = dLdA * dAdZ

        # Gradient wrt the learnable beta parameter (summed over the batch)
        dAdbeta = (self.Z ** 2) * self.sig * (1 - self.sig)
        self.dLdbeta = np.sum(dLdA * dAdbeta)

        return dLdZ


class Softmax:
    """
    Softmax activation function.
    """

    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        """
        row_maxes = np.max(Z, axis=1, keepdims=True)
        normalized = np.exp(Z - row_maxes)
        combined = (np.ones((1, Z.shape[1])) @ normalized.T).T  # create 1xCin matrix of ones, multiply by exponentiated Z to get combined values for each row
        # combined is now Nx1 where each entry is the sum of the exponentiated values for that row
        self.A = normalized / combined
        return self.A

    def backward(self, dLdA):
        # Calculate the batch size and number of features
        N = self.A.shape[0]
        C = self.A.shape[1]

        # Initialize the final output dLdZ with all zeros.
        dLdZ = np.zeros((N, C))

        # Fill dLdZ one data point (row) at a time.
        for i in range(N):
            # Initialize the Jacobian with all zeros.
            J = np.zeros((C, C))

            # Fill the Jacobian matrix, please read the writeup for the conditions.
            for m in range(C):
                for n in range(C):
                    if (m == n):
                        J[m, n] = self.A[i, m] * (1 - self.A[i, m])
                    else:
                        J[m, n] = -self.A[i, n] * self.A[i, m]

            # Calculate the derivative of the loss with respect to the i-th input.
            dLdZ[i] = dLdA[i] @ J

        return dLdZ