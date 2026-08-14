import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        out_size  = (A.shape[2]-self.kernel_size)+1
        Z = np.zeros((A.shape[0], self.out_channels, out_size))

        for i in range(out_size): #will go from (0,output_size)
            #A = (batch, num in channels, input size)
            # W = (out_channels, in_channels, kernel size)
            # we slice A to just get (kenerl_size) of the input each time
            #
            Z[:,:,i] = np.tensordot(A[:,:,i:self.kernel_size+i], self.W, axes=([1,2],[1,2]))

        return Z+self.b[:, None] #need to add the extra dimensions so broadcasting works as intended

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # dLdW: gradient must be shaped like W: (C_out, C_in, K).
        #   Convolve the FULL input A (width W_in) using dLdZ as the kernel (width W_out).
        #   Result has width W_in - W_out + 1 = K. SUM over the batch axis (do NOT average).

        # dLdA: gradient must be shaped like A: (N, C_in, W_in).
        #   Pad dLdZ with (K-1) zeros on each side, FLIP each filter left-to-right,
        #   then convolve. SUM the contributions over the output-channel axis C_out.
        #   Batch axis is preserved — no averaging anywhere.

        Padded_derivative = np.pad(dLdZ, pad_width = ((0,0),(0,0),(self.kernel_size - 1, self.kernel_size - 1)))
        Flipped_weights = self.W[:,:,::-1] #reverses the orientatin (if we treat each row as indpendent array, reverse each array)

        N = dLdZ.shape[0]
        in_size = dLdZ.shape[2]-1+self.kernel_size

        self.dLdb = np.sum(dLdZ, axis = (0,2))  # one scalar value per out_channel. Sum along other dimensions
        dLdA = np.zeros((N, self.in_channels, in_size))  # TODO


        for i in range(self.kernel_size): #will go from (0,K)
            self.dLdW[:,:,i] = np.tensordot(dLdZ, self.A[:,:,i:dLdZ.shape[2]+i], axes=([0,2],[0,2]))

        for i in range(in_size):
            dLdA[:,:,i] = np.tensordot(Padded_derivative[:,:,i:self.kernel_size+i], Flipped_weights, axes=([1,2],[0,2]))

        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        self.stride = stride
        self.pad = padding
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size,
                                             weight_init_fn, bias_init_fn)
        self.downsample1d = Downsample1d(stride)

    def forward(self, A):
        Padded = np.pad(A, ((0,0), (0,0), (self.pad, self.pad)))
        Convolved = self.conv1d_stride1.forward(Padded)
        Z = self.downsample1d.forward(Convolved)
        return Z

    def backward(self, dLdZ):
        dLdZ_up = self.downsample1d.backward(dLdZ)
        dLdA_padded = self.conv1d_stride1.backward(dLdZ_up)
        if self.pad > 0:
            dLdA = dLdA_padded[:, :, self.pad:-self.pad]
        else:
            dLdA = dLdA_padded
        return dLdA