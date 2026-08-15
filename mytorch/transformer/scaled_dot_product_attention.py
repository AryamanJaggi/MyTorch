import numpy as np
from mytorch.nn.activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        self.eps = 1e10
        self.softmax = Softmax(dim=-1) #we want to compute softmax per instance, so across embedding dimension
        
    
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E) where L is target sequence length
        :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
        :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S) or broadcastable shape where 1/True indicates a position to ignore
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """

        self.V = V
        self.Q = Q
        self.K = K
        
        # Calculate attention scores
        scaled_dot_product = (Q@np.swapaxes(self.K, -1, -2))/np.sqrt(Q.shape[-1])
        
        # Apply mask before softmax if provided
        if mask is not None:
            scaled_dot_product = scaled_dot_product-(mask*self.eps)

        # Compute attention scores: 
        # # Think about which dimension you should apply Softmax
        self.attention_scores = self.softmax.forward(scaled_dot_product)

        # Calculate final output
        output = self.attention_scores@V

        # Return final output
        return output
    
    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """

        # Calculate gradients for V
        d_V = (np.swapaxes(self.attention_scores, -1, -2))@d_output
        
        # Calculate gradients for attention scores
        d_attention_scores = d_output @ (np.swapaxes(self.V, -1, -2))
        d_scaled_dot_product = self.softmax.backward(d_attention_scores)
        
        # Scale gradients by sqrt(d_k)
        d_dot_product = d_scaled_dot_product/np.sqrt(self.Q.shape[-1])
        
        # Calculate gradients for Q and K
        d_Q = d_dot_product @ self.K
        d_K = np.swapaxes(d_dot_product, -1, -2) @ self.Q
        
        # Return gradients for Q, K, V
        return d_Q,d_K,d_V