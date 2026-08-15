import numpy as np

'''
Specification:
- Function creates a padding mask that identifies padded positions in the input
- Mask is a boolean array of shape (N, T) where:
  * N = batch size from padded_input
  * T = sequence length from padded_input
- True values indicate padding positions that should be masked
- False values indicate valid positions that should not be masked
- Padding is assumed to be on the right side of sequences
- Each sequence in the batch may have different valid lengths
'''
def PadMask(padded_input, input_lengths):
    """
    Create a mask for padding positions.
    Args:
        padded_input: The input array, shape (N, T, ...).
        input_lengths: Actual lengths before padding, shape (N,).
    Returns:
        Boolean mask array with shape (N, T).
    """
    N = padded_input.shape[0]
    T = padded_input.shape[1]

    # row of positions [0, 1, ..., T-1], shape (T,)
    positions = np.arange(T)

    lengths = np.asarray(input_lengths)

    # broadcast: (1, T) >= (N, 1)  ->  (N, T)
    # True where position index >= that sequence's true length -> padding
    mask = positions[np.newaxis, :] >= lengths[:, np.newaxis]

    return mask


'''
Specification:
- Function creates a causal mask for self-attention
- Mask is a boolean array of shape (T, T) where T is sequence length
- True values indicate positions that should not attend to each other
- False values indicate positions that can attend to each other
- Causal means each position can only attend to itself and previous positions
- Mask should be upper triangular (excluding diagonal)
'''
def CausalMask(padded_input):
    """
    Create a causal mask for self-attention.
    Args:
        padded_input: Input array, shape (N, T, ...).
    Returns:
        Boolean mask array with shape (T, T).
    """
    T = padded_input.shape[1]
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    return mask