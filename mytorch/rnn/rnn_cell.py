import numpy as np
from mytorch.nn.activation import *


class RNNCell(object):
    """RNN Cell class."""

    def __init__(self, input_size, hidden_size):

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Activation function for
        self.activation = Tanh()

        # hidden dimension and input dimension
        h = self.hidden_size
        d = self.input_size

        # Weights and biases
        self.W_ih = np.random.randn(h, d)
        self.W_hh = np.random.randn(h, h)
        self.b_ih = np.random.randn(h)
        self.b_hh = np.random.randn(h)

        # Gradients
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))

        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def init_weights(self, W_ih, W_hh, b_ih, b_hh):
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.b_ih = b_ih
        self.b_hh = b_hh

    def zero_grad(self):
        d = self.input_size
        h = self.hidden_size
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))
        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """
        RNN Cell forward (single time step).
        -----
        Input (see writeup for explanation)
        x: (batch_size, input_size)

        h_prev_t: (batch_size, hidden_size)
        -----
        Returns
        h_t: (batch_size, hidden_size)
            hidden state at the current time step and current layer
        """
    
        """ ht = tanh(Wihxt + bih + Whhht−1 + bhh) """

        # transformed x. Result of matmul is NxH_out. Bias is H_out. Will be broadcasted
        t_x = x@self.W_ih.T + self.b_ih
        
        #transformed previous hidden state
        t_h_prev_t = h_prev_t@self.W_hh.T + self.b_hh

        h_t = self.activation.forward(t_x+t_h_prev_t)

        # return h_t
        return h_t

    def backward(self, delta, h_t, h_prev_l, h_prev_t):
        """
        RNN Cell backward (single time step).

        -----
        Input (see writeup for explanation)
        
        delta: (batch_size, hidden_size)
        
        h_t: (batch_size, hidden_size)

        h_prev_l: (batch_size, input_size)

        h_prev_t: (batch_size, hidden_size)

        ------
        Returns

        dx: (batch_size, input_size)
            Derivative w.r.t.  the current time step and previous layer

        dh_prev_t: (batch_size, hidden_size)
            Derivative w.r.t.  the previous time step and current layer

        """
        batch_size = delta.shape[0]
        # 0) Done! Step backward through the tanh activation function.
        dz = self.activation.backward(delta)  # TODO

        # 1) Compute the averaged gradients of the weights and biases
        self.dW_ih += (dz.T@h_prev_l)/batch_size # divide by batch size to average influence across training instances
        self.dW_hh += (dz.T@h_prev_t)/batch_size  # TODO
        self.db_ih += np.sum(dz, axis=0)/batch_size  # TODO
        self.db_hh += np.sum(dz, axis=0)/batch_size  # TODO

        # 2) Compute dx, dh_prev_t
        dx = dz@self.W_ih  # TODO
        dh_prev_t = dz@self.W_hh  # TODO

        # 3) Return dx, dh_prev_t
        return dx, dh_prev_t
