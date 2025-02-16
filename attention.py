import torch
import torch.nn as nn
from xformers.ops import memory_efficient_attention
import os

class XFormersAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, layer_id=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "Embedding dim must be divisible by num_heads"

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Learnable weight for scaling attention outputs before passing to the next block
        self.weight = nn.Parameter(torch.ones(1))

        # Cache storage for autoregressive inference
        self.cached_keys = None
        self.cached_values = None

        # Unique identifier for each layer
        self.layer_id = layer_id

    def forward(self, query, key, value, use_cache=False):
        batch_size, seq_len, _ = query.shape

        # Apply Layer Normalization before attention
        query = self.layer_norm(query)
        key = self.layer_norm(key)
        value = self.layer_norm(value)

        # Reshape into multi-head format
        Q = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Use cached keys/values for incremental decoding
        if use_cache:
            if self.cached_keys is not None and self.cached_values is not None:
                K = torch.cat([self.cached_keys, K], dim=2)
                V = torch.cat([self.cached_values, V], dim=2)
            self.cached_keys = K.detach()
            self.cached_values = V.detach()

        # Apply xFormers' memory-efficient attention
        attn_output = memory_efficient_attention(Q, K, V, attn_bias=None)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Apply output projection
        output = self.out_proj(attn_output)

        # Scale output using the learnable weight before passing to the next block
        output = output * self.weight

        # Save weights to file
        self.save_weights()

        return output, self.cached_keys, self.cached_values

    def save_weights(self):
        """Saves the current attention weights to a file."""
        save_path = f"attention_weights/layer_{self.layer_id}.pth"
        os.makedirs("attention_weights", exist_ok=True)
        torch.save(self.state_dict(), save_path)
