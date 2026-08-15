import numpy as np


class Upsample1d():
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        W_out = (A.shape[2] - 1) * self.upsampling_factor + 1
        Z = np.zeros((A.shape[0], A.shape[1], W_out))
        Z[:,:,::self.upsampling_factor] = A  # TODO
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        dLdA = dLdZ[:,:,::self.upsampling_factor]  # TODO
        return dLdA


class Downsample1d():
    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        self.w_in = A.shape[2]
        Z = A[:,:,::self.downsampling_factor]  # TODO
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], self.w_in))  # TODO
        dLdA[:,:,::self.downsampling_factor] = dLdZ
        return dLdA


class Upsample2d():
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        N,C,H,W = A.shape
        w_out = ((H-1)*self.upsampling_factor+1, (W-1)*self.upsampling_factor+1)
        Z = np.zeros((N,C, w_out[0], w_out[1]))
        Z[:,:,::self.upsampling_factor, ::self.upsampling_factor] = A  # TODO
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        dLdA = dLdZ[:,:,::self.upsampling_factor, ::self.upsampling_factor]  # TODO
        return dLdA


class Downsample2d():
    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        N,C,self.H_in,self.W_in = A.shape
        Z = A[:,:,::self.downsampling_factor, ::self.downsampling_factor]  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], self.H_in, self.W_in))  # FIXED: was missing the tuple parens
        dLdA[:,:,::self.downsampling_factor, ::self.downsampling_factor] = dLdZ
        return dLdA
