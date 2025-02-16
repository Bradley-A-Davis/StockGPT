import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=98_280):
        super().__init__()
        self.embed_dim = embed_dim

        # Create a matrix of shape (max_seq_len, embed_dim)
        position = torch.arange(max_seq_len).unsqueeze(1)  # Shape: (max_seq_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))

        pe = torch.zeros(max_seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        
        self.register_buffer('pe', pe)  # Ensures it does not update during training

    def forward(self, x):
        """
        Adds positional encoding to the input embeddings.
        x: Tensor of shape (batch_size, seq_len, embed_dim)
        """
        seq_len = x.shape[1]
        x = x + self.pe[:seq_len, :].unsqueeze(0)  # Add positional encodings
        return x
