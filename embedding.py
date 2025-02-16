import torch
import torch.nn as nn
import math
import os

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
        
        self.register_buffer('pe', pe, persistent=False)  # Ensures it does not update during training

    def forward(self, x):
        """
        Adds positional encoding to the input embeddings.
        x: Tensor of shape (batch_size, seq_len, embed_dim)
        """
        seq_len = x.shape[1]
        x = x + self.pe[:seq_len, :].unsqueeze(0)  # Add positional encodings
        return x

# Restore original embedding logic
start, end, step = -5, 5, 0.01
values = torch.arange(start, end + step, step)  # 1001 values
num_embeddings = len(values)  # 1001 unique values
embedding_dim = 1000  # Each token maps to a 1000-dimensional vector

embedding_layer = nn.Embedding(num_embeddings, embedding_dim)

# Load saved weights if available
if os.path.exists("embedding_weights.pth"):
    embedding_layer.load_state_dict(torch.load("embedding_weights.pth", weights_only=True))

def get_tokenized_input(batch_size, seq_len):
    """ Generate random sequences of token indices as input. """
    return torch.randint(0, num_embeddings, (batch_size, seq_len))

def embed_input(token_indices):
    """ Convert token indices into embeddings. """
    return embedding_layer(token_indices)  # Shape: (batch_size, seq_len, embed_dim)
