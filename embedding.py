import torch
import torch.nn as nn
import numpy as np
import os

# Define embedding range and mapping
start, end, step = -5, 5, 0.01
values = np.arange(start, end + step, step)  # 1001 values

num_embeddings = len(values)  # 1001 unique values
embedding_dim = 1000  # Each token maps to a 1000-dimensional vector

# Define the embedding layer
embedding_layer = nn.Embedding(num_embeddings, embedding_dim)

# Load saved weights if available
if os.path.exists("embedding_weights.pth"):
    embedding_layer.load_state_dict(torch.load("embedding_weights.pth"))

def get_tokenized_input(batch_size, seq_len):
    """ Generate random sequences of token indices as input. """
    return torch.randint(0, num_embeddings, (batch_size, seq_len))

def embed_input(token_indices):
    """ Convert token indices into embeddings. """
    return embedding_layer(token_indices)  # Shape: (batch_size, seq_len, embed_dim)