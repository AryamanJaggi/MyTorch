import numpy as np


class MSELoss:
    def forward(self, A, Y):
        """
        Calculate the Mean Squared error (MSE)
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss (scalar)

        """
        self.A = A
        self.Y = Y
        self.N = A.shape[0]
        self.C = A.shape[1]
        se = (A-Y)*(A-Y)
        sse = np.sum(se)
        mse = sse/(self.N*self.C)
        return mse

    def backward(self):
        """
        Calculate the gradient of MSE Loss wrt model output A.
        :Return: Gradient of loss L wrt model output A.

        """
        dLdA = 2*(self.A - self.Y)/(self.N*self.C)
        return dLdA


class CrossEntropyLoss:
    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss (XENT)
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss (scalar)

        """
        self.A = A
        self.Y = Y
        self.N = A.shape[0]
        self.C = A.shape[1]

        Ones_C = np.ones((self.C, 1), dtype='f')
        Ones_N = np.ones((self.N, 1), dtype='f')

        # Numerically stable softmax
        row_maxes = np.max(A, axis=1, keepdims=True)
        exp_A = np.exp(A - row_maxes)
        self.softmax = exp_A / (exp_A @ Ones_C)

        crossentropy = (-Y * np.log(self.softmax)) @ Ones_C
        sum_crossentropy_loss = (Ones_N.T @ crossentropy).item()
        mean_crossentropy_loss = sum_crossentropy_loss / self.N

        return mean_crossentropy_loss

    def backward(self):
        """
        Calculate the gradient of Cross-Entropy Loss wrt model output A.
        :Return: Gradient of loss L wrt model output A.
        
        """
        dLdA = self.softmax - self.Y
        return dLdA