import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during inference.
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        Forward pass for batch normalization.
        :param Z: batch of input data Z (N, num_features).
        :param eval: flag to indicate training or inference mode.
        :return: batch normalized data.

        """
        self.Z = Z
        self.N = Z.shape[0]  # Calculate batch size
        self.M = np.sum(Z, axis=0, keepdims=True)/self.N  # Calculate mini-batch mean. Axis 0 is the rows. This dimensions will get deleted
        #without keepdims would just have shape (C), 1 dim array. With keepdims it stays a matrix, 1xC
        dif = self.M-Z #broadcasting rules will apply. Now every element of diff holds E[x]-x where E[x] is the mean of that feature across all batch sample
        square = dif*dif # now each entry holds (E[x]-x)^2
        self.V = np.sum(square, axis=0, keepdims=True)/self.N  # Calculate mini-batch variance
        #Now V is 1xC and holds the variance for each paramater

        if eval == False:
            # training mode
            self.NZ = (Z-self.M)/np.sqrt((self.V + self.eps))  # Calculate the normalized input Ẑ
            self.BZ = self.NZ*self.BW+self.Bb  # Calculate the scaled and shifted for the normalized input Ẑ

            self.running_M = self.alpha*self.running_M + (1-self.alpha)*self.M  # Calculate running mean
            self.running_V = self.alpha*self.running_V + (1-self.alpha)*self.V   # Calculate running variance
        else:
            # inference mode
            self.NZ = (Z-self.running_M)/np.sqrt((self.running_V + self.eps))  # Calculate the normalized input Ẑ using the running average for mean and variance
            self.BZ = self.NZ*self.BW+self.Bb  # Calculate the scaled and shifted for the normalized input Ẑ

        return self.BZ

    def backward(self, dLdBZ):
        """
        Backward pass for batch normalization.
        :param dLdBZ: Gradient loss wrt the output of BatchNorm transformation for Z (N, num_features).
        :return: Gradient of loss (L) wrt batch of input batch data Z (N, num_features).

        """
        self.dLdBb = np.sum(dLdBZ, axis=0, keepdims=True)  # Sum over the batch dimension. Result is 1xC
        self.dLdBW = np.sum(dLdBZ*self.NZ, axis=0, keepdims=True)   # Scale gradient of loss wrt BatchNorm transformation by normalized input NZ.

        dLdNZ = dLdBZ*self.BW  # Scale gradient of loss wrt BatchNorm transformation output by gamma (scaling parameter).

        dLdV = np.sum(dLdNZ*(-0.5)*np.power(self.V + self.eps, -1.5)*(self.Z - self.M), axis=0, keepdims=True)  # Compute gradient of loss backprop through variance calculation.
        dLdM = np.sum(dLdNZ*(-1/np.sqrt(self.V + self.eps)), axis=0, keepdims=True) \
               + dLdV*(-2/self.N)*np.sum(self.Z - self.M, axis=0, keepdims=True)  # Compute gradient of loss with respect to mean.

        dLdZ = dLdNZ/np.sqrt(self.V + self.eps) + dLdV*2*(self.Z - self.M)/self.N + dLdM/self.N  # Compute gradient of loss with respect to the input.
        return dLdZ