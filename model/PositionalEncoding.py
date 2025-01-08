import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional Encoder for the Transformer model.
    """
    def __init__(self, embedding_dim, max_len=2048):
        super().__init__()
        self.embed = embedding_dim
        self.pe = None
        self.pe = torch.zeros(max_len, embedding_dim)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))  # (d_model / 2)
        # Apply sine to even indices
        self.pe[:, 0::2] = torch.sin(position * div_term)  # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        self.pe[:, 1::2] = torch.cos(position * div_term)  # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        # pe = self.pe.unsqueeze(0)  # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        # self.register_buffer('pe', self.pe)
        # self.register_buffer('pe', self.pe)




    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, embedding_dim]
        """

        # print(self.pe.shape)
        return self.pe.unsqueeze(0).requires_grad_(False)
