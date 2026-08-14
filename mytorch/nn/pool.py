import numpy as np
from resampling import *


class MaxPool2d_stride1():
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        N, C, H_in, W_in = A.shape
        H_out = H_in-self.kernel+1
        W_out = W_in-self.kernel+1
        Z = np.zeros((N,C,H_out, W_out))
        self.indices = np.zeros((N,C,H_out, W_out), dtype=int)
        for i in range(H_out):
            for j in range(W_out):
                window = A[:,:,i:i+self.kernel, j:j+self.kernel].reshape(N, C, -1)
                idx = np.argmax(window, axis=-1)
                self.indices[:,:,i,j] = idx
                Z[:,:,i,j] = np.take_along_axis(window, idx[..., None], axis=-1).squeeze(-1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        N, C, H_out, W_out = dLdZ.shape
        H_in = H_out+self.kernel-1
        W_in = W_out+self.kernel-1
        dLdA = np.zeros((N, C, H_in, W_in))
        n_idx, c_idx = np.meshgrid(np.arange(N), np.arange(C), indexing='ij')
        for i in range(H_out):
            for j in range(W_out):
                idx = self.indices[:,:,i,j]
                di = idx // self.kernel
                dj = idx % self.kernel
                np.add.at(dLdA, (n_idx, c_idx, i+di, j+dj), dLdZ[:,:,i,j])

        return dLdA


class MeanPool2d_stride1():
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        N, C, H_in, W_in = A.shape
        H_out = H_in-self.kernel+1
        W_out = W_in-self.kernel+1
        Z = np.zeros((N, C, H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                Z[:,:,i,j] = np.mean(A[:,:,i:i+self.kernel, j:j+self.kernel], axis=(2,3))

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        N, C, H_out, W_out = dLdZ.shape
        H_in = H_out+self.kernel-1
        W_in = W_out+self.kernel-1
        dLdA = np.zeros((N, C, H_in, W_in))
        for i in range(H_out):
            for j in range(W_out):
                dLdA[:,:,i:i+self.kernel, j:j+self.kernel] += dLdZ[:,:,i,j][:,:,None,None] / (self.kernel**2)

        return dLdA


class MaxPool2d():
    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.downsample2d.forward(self.maxpool2d_stride1.forward(A))

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = self.maxpool2d_stride1.backward(self.downsample2d.backward(dLdZ))

        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.downsample2d.forward(self.meanpool2d_stride1.forward(A))

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = self.meanpool2d_stride1.backward(self.downsample2d.backward(dLdZ))

        return dLdA