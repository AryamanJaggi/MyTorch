import numpy as np
from .resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        N, _, H_in, W_in = A.shape

        H_out = H_in-self.kernel_size+1
        W_out = W_in-self.kernel_size+1

        Z = np.zeros((N, self.out_channels, H_out, W_out))

        for i in range(H_out):
            for j in range(W_out):
                Z[:,:,i,j] = np.tensordot(A[:,:,i:i+self.kernel_size, j:j+self.kernel_size], self.W, axes=([1,2,3], [1,2,3]))

        return Z+self.b[:,None,None] #rigth pad for broadcasting. Numpy handles left pad.

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        N, _, H_out, W_out = dLdZ.shape
        H_in = H_out+self.kernel_size-1
        W_in = W_out+self.kernel_size-1
        Padded = np.pad(dLdZ, ((0,0), (0,0), (self.kernel_size-1, self.kernel_size-1), (self.kernel_size-1, self.kernel_size-1)))
        Reversed_weight = self.W[:,:,::-1,::-1]
        dLdA = np.zeros((N, self.in_channels, H_in, W_in))


        for i in range (self.kernel_size):
            for j in range (self.kernel_size):
                self.dLdW[:,:,i,j] = np.tensordot(dLdZ, self.A[:,:,i:i+H_out, j:j+W_out], axes=([0,2,3], [0,2,3]))
        
        for i in range (H_in):
            for j in range (W_in):
                dLdA[:,:,i,j] = np.tensordot(Padded[:,:,i:i+self.kernel_size, j:j+self.kernel_size], Reversed_weight, axes=([1,2,3], [0,2,3]))

        self.dLdb = np.sum(dLdZ, axis = (0,2,3))

        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size,
                                              weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # Pad the input appropriately using np.pad() function
        Padded = np.pad(A, ((0,0), (0,0), (self.pad, self.pad), (self.pad, self.pad)))

        # Call Conv2d_stride1
        Convolved = self.conv2d_stride1.forward(Padded)

        # downsample
        Z = self.downsample2d.forward(Convolved)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        # Call downsample2d backward
        dLdZ_up = self.downsample2d.backward(dLdZ)

        # Call Conv2d_stride1 backward
        dLdA_padded = self.conv2d_stride1.backward(dLdZ_up)

        # Unpad the gradient
        if self.pad > 0:
            dLdA = dLdA_padded[:, :, self.pad:-self.pad, self.pad:-self.pad]
        else:
            dLdA = dLdA_padded

        return dLdA 