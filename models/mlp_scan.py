from mytorch.nn.flatten import *
from mytorch.nn.Conv1d import *
from mytorch.nn.linear import *
from mytorch.nn.activation import *
from mytorch.nn.loss import *
import numpy as np
import os
import sys

class CNN_SimpleScanningMLP():
    def __init__(self):
        self.conv1 = Conv1d(24,8,8,4) #24 input channels, 8 output channels. Scans 8 time-steps at a time, so kernel size 8. Stride 4
        self.relu1 = ReLU()
        self.conv2 = Conv1d(8,16,1,1)
        self.relu2 = ReLU()
        self.conv3 = Conv1d(16,4,1,1)
        self.flatten = Flatten()
        self.layers = [self.conv1, self.relu1, self.conv2, self.relu2, self.conv3, self.flatten]

    def init_weights(self, weights):
        """
        Args:
            w1 (np.array): (kernel_size * in_channels, out_channels)
            w2 (np.array): (kernel_size * in_channels, out_channels)
            w3 (np.array): (kernel_size * in_channels, out_channels)
        """
        w1, w2, w3 = weights[0], weights[1], weights[2]
        w1t = w1.T
        w2t = w2.T
        w3t = w3.T

        w1r = w1t.reshape(w1t.shape[0], 8, 24)
        w2r = w2t.reshape(w2t.shape[0], 1, 8)
        w3r = w3t.reshape(w3t.shape[0], 1, 16)

        self.conv1.conv1d_stride1.W = w1r.transpose(0, 2, 1)
        self.conv2.conv1d_stride1.W = w2r.transpose(0, 2, 1)
        self.conv3.conv1d_stride1.W = w3r.transpose(0, 2, 1)

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """
        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method
        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA


class CNN_DistributedScanningMLP():
    def __init__(self):
        self.conv1 = Conv1d(24,2,2,2) #24 input channels, 2 output channels. Scans 2 time-steps at a time, so kernel size 2. Stride 2
        self.relu1 = ReLU()
        self.conv2 = Conv1d(2,8,2,2) #2 input channels, 8 output channels. Scans 2 time steps at time. Stride 2
        self.relu2 = ReLU()
        self.conv3 = Conv1d(8,4,2,1) #8 inpur channels, 4 output channels. Scans 2 steps at a time. Stride 1
        self.flatten = Flatten()
        self.layers = [self.conv1, self.relu1, self.conv2, self.relu2, self.conv3, self.flatten]

    def __call__(self, A):
        # Do not modify this method
        return self.forward(A)

    def init_weights(self, weights):
        """
        Args:
            w1 (np.array): (kernel_size * in_channels, out_channels)
            w2 (np.array): (kernel_size * in_channels, out_channels)
            w3 (np.array): (kernel_size * in_channels, out_channels)
        """
        w1, w2, w3 = weights[0], weights[1], weights[2]

        w1 = w1[:48, :2]
        w2 = w2[:4, :8]
        w3 = w3[:16, :4]

        w1t = w1.T
        w2t = w2.T
        w3t = w3.T

        temp1 = w1t.reshape(2, 2, 24).transpose(0, 2, 1)
        temp2 = w2t.reshape(8, 2, 2).transpose(0, 2, 1)
        temp3 = w3t.reshape(4, 2, 8).transpose(0, 2, 1)

        self.conv1.conv1d_stride1.W = temp1
        self.conv2.conv1d_stride1.W = temp2
        self.conv3.conv1d_stride1.W = temp3

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """
        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA