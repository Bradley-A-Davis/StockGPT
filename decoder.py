import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTDecoder(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.proj = nn.Linear(embed_dim, vocab_size)  # Map embedding to vocab
        self.log_softmax = nn.LogSoftmax(dim=-1)  # Converts logits to probabilities

    def forward(self, x):
        """
        x: (batch_size, seq_len, embed_dim) - The processed embeddings from GPT
        Returns: (batch_size, seq_len, vocab_size) - Logits over vocabulary
        """
        logits = self.proj(x)  # Convert embeddings to vocab logits
        return logits