import numpy as np


class PositionalEncoding:
    def __init__(self, d_model, max_len):
        """
        Initialize the PositionalEncoding.
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.
        """
        self.create_pe_table(d_model, max_len)

    def create_pe_table(self, d_model, max_len):
        """
        Create the positional encoding table.
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.
        """
        pe = np.zeros((max_len, d_model))

        position = np.arange(max_len)[:, np.newaxis]              # (max_len, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))  # (d_model/2,)

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        # Store with a leading batch dim of 1 so it broadcasts over (B, T, d_model)
        self.pe = pe[np.newaxis, :, :]   # (1, max_len, d_model)

    def forward(self, x):
        """
        Forward pass for the PositionalEncoding.
        Args:
            x (np.ndarray): The input array of shape (B, T, d_model)
        Returns:
            np.ndarray: Input with positional encoding added (B, T, d_model)
        Errors:
            - ValueError: If sequence length exceeds maximum length
        """
        seq_len = x.shape[1]

        if seq_len > self.pe.shape[1]:
            raise ValueError(f"Sequence length {seq_len} exceeds the maximum length {self.pe.shape[1]}")

        return x + self.pe[:, :seq_len, :]

    def __call__(self, x):
        return self.forward(x)