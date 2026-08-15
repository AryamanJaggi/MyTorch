import numpy as np
import sys

from mytorch.rnn.rnn_cell import *
from mytorch.nn.linear import *


class RNNPhonemeClassifier(object):
    """RNN Phoneme Classifier class."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        
        self.rnn = [
            RNNCell(input_size, hidden_size) if i == 0 
                else RNNCell(hidden_size, hidden_size)
                    for i in range(num_layers)
        ]
        self.output_layer = Linear(hidden_size, output_size)

        self.hiddens = []

    def init_weights(self, rnn_weights, linear_weights):
        """Initialize weights.
        -----
        Input
        rnn_weights:
                    [
                        [W_ih_l0, W_hh_l0, b_ih_l0, b_hh_l0],
                        [W_ih_l1, W_hh_l1, b_ih_l1, b_hh_l1],
                        ...
                    ]
        linear_weights:
                        [W, b]
        """
        for i, rnn_cell in enumerate(self.rnn):
            rnn_cell.init_weights(*rnn_weights[i])
        self.output_layer.W = linear_weights[0]
        self.output_layer.b = linear_weights[1].reshape(-1, 1)

    def __call__(self, x, h_0=None):
        return self.forward(x, h_0)

    def forward(self, x, h_0=None):
        """RNN forward, multiple layers, multiple time steps.
        -----
        Input (see writeup for explanation)
        x: (batch_size, seq_len, input_size)

        h_0: (num_layers, batch_size, hidden_size)
            Initial hidden states - defaults to zeros if not specified
        -------
        Returns
        logits: (batch_size, output_size) 

        Output (y): logits

        """
        # Get the batch size and sequence length, and initialize the hidden
        # vectors given the paramters.
        batch_size, seq_len = x.shape[0], x.shape[1]
        if h_0 is None:
            hidden = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        else:
            hidden = h_0

        # Reset hiddens at the start of every forward call. Without this,
        # self.hiddens keeps growing across calls and all the index
        # arithmetic in backward() (hiddens[i], hiddens[i+1], hiddens[seq_len])
        # silently points at stale timesteps from a previous forward pass.
        self.hiddens = []

        # Save x and append the hidden vector to the hiddens list
        self.x = x
        self.hiddens.append(hidden.copy())
        logits = None

        # TODO
        for i in range(seq_len):
            self.hiddens.append(hidden.copy())
            for j, rnn_cell in enumerate(self.rnn):
                if (j == 0):
                    (self.hiddens[i+1])[j,:,:] = rnn_cell.forward(self.x[:,i,:], (self.hiddens[i])[j,:,:])
                else:
                    (self.hiddens[i+1])[j,:,:] = rnn_cell.forward((self.hiddens[i+1])[j-1,:,:], (self.hiddens[i])[j,:,:])

        # Get the outputs from the last time step using the linear layer and return it
        #there are seq_length+1 hidden values because of the initial value
        logits = self.output_layer.forward((self.hiddens[seq_len])[self.num_layers-1, :, :])
        
        # return logits
        return logits

    def backward(self, delta):
        """RNN Back Propagation Through Time (BPTT).

        Input (see writeup for explanation)
        ------
        delta: (batch_size, hidden_size)

        gradient: dY(seq_len-1)
                
        Returns
        -------
        dh_0: (num_layers, batch_size, hidden_size)

        gradient w.r.t. the initial hidden states

        """
        # Initilizations
        batch_size, seq_len = self.x.shape[0], self.x.shape[1]
        dh = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        dh[-1] = self.output_layer.backward(delta)

        for i in range(seq_len-1, -1, -1):
            dx_above = None
            for j in range (self.num_layers-1, -1, -1):

                #need to deal with last layer differently. If it's also the last sequence, then we propogate derivative wrt to input 
                #to linear layer. If it's not last sequence, then because we don't make outputs for other sequences, we only propogate
                #gradient wrt to input to next sequence's same hidden layer
                if (j == self.num_layers-1): #last layer
                    if (i == seq_len-1): #last sequence
                        dx_above, dh[j,:,:] = self.rnn[j].backward(dh[j], (self.hiddens[i+1])[j, :, :], (self.hiddens[i+1])[j-1, :, :], (self.hiddens[i])[j, :, :]) ##add 1 to i because hiddens has en entry for initial hidden state as well
                    else:
                        dx_above, dh[j,:,:] = self.rnn[j].backward(dh[j], (self.hiddens[i+1])[j, :, :], (self.hiddens[i+1])[j-1, :, :], (self.hiddens[i])[j, :, :])

                #first layer also needs to be handled differently because input is not previous hidden layer's output(there is none)
                #input is X[:,t,:] 
                elif (j == 0): #first layer
                    if (i == seq_len-1): #last sequence
                        dx_above, dh[j,:,:] = self.rnn[j].backward(dx_above, (self.hiddens[i+1])[j, :, :], self.x[:,i,:], (self.hiddens[i])[j, :, :])
                    else:
                        dx_above, dh[j,:,:] = self.rnn[j].backward(dh[j]+dx_above, (self.hiddens[i+1])[j, :, :], self.x[:,i,:], (self.hiddens[i])[j, :, :])
                
                else: #middle layers
                    if (i == seq_len-1): #last sequence doesn't get influenced by other sequences
                        dx_above, dh[j,:,:] = self.rnn[j].backward(dx_above, (self.hiddens[i+1])[j, :, :], (self.hiddens[i+1])[j-1, :, :], (self.hiddens[i])[j, :, :])
                    else:#at this point, dh[i] has been updates, so it encodes how input to next layer of this sequence affects loss
                        #dh[i-1] has not been updates, it encodes how input to this layer in next sequence affects loss
                        dx_above, dh[j,:,:] = self.rnn[j].backward(dh[j]+dx_above, (self.hiddens[i+1])[j, :, :], (self.hiddens[i+1])[j-1, :, :], (self.hiddens[i])[j, :, :])

        
        # TODO

        # dh_0 must keep the same shape as h_0: (num_layers, batch_size, hidden_size).
        # Only divide by batch_size
        return dh / batch_size