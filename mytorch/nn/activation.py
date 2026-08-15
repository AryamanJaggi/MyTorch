import numpy as np
import scipy.special

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
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # 1. Shift Z for numerical stability
        # Keepdims=True ensures it broadcasts correctly during subtraction
        Z_shifted = Z - np.max(Z, axis=self.dim, keepdims=True)
    
        exponented = np.exp(Z_shifted)
        summed = np.sum(exponented, axis=self.dim, keepdims = True)

        self.A = exponented/summed
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        #reshape
        dLdA = np.moveaxis(dLdA, self.dim, -1)
        self.A = np.moveaxis(self.A, self.dim, -1)

        #save input shape
        input_shape = self.A.shape

        #flatten
        dLdA = dLdA.reshape(-1,dLdA.shape[-1])
        self.A = self.A.reshape(-1,self.A.shape[-1])

        # Calculate the * size and number of features
        N = self.A.shape[0]  # TODO
        C = self.A.shape[1]  # TODO

        # Initialize the final output dLdZ with all zeros.
        dLdZ = np.zeros((N,C))  # TODO

        # Fill dLdZ one data point (row) at a time.
        for i in range(N):
            # Initialize the Jacobian with all zeros.
            J = np.zeros((C,C))  # TODO

            # Fill the Jacobian matrix
            for m in range(C):
                for n in range(C):
                    if (m == n): 
                        J[m, n] = self.A[i,m]*(1 - self.A[i,m])
                    else:
                        J[m, n] = -self.A[i,n]*self.A[i,m]

            dLdZ[i] = dLdA[i]@J
        
        #unflatten
        dLdZ = dLdZ.reshape(input_shape)
        self.A = self.A.reshape(input_shape)

        #swap back
        dLdZ = np.moveaxis(dLdZ, -1, self.dim)
        self.A = np.moveaxis(self.A, -1, self.dim)

        return dLdZ