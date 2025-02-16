import torch
import torch.nn as nn
from embedding import embed_input, get_tokenized_input
from positional_encoding import PositionalEncoding
from decoder import GPTDecoder
from attention import XFormersAttentionBlock

class GPTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, layer_id=0):
        super().__init__()
        self.attention = XFormersAttentionBlock(embed_dim, num_heads, dropout, layer_id)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        # Learnable weight to scale attention output before passing to the next layer
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x, use_cache=False):
        attn_out, cached_k, cached_v = self.attention(x, x, x, use_cache=use_cache)
        x = self.layer_norm1(attn_out + x)  # Add & Norm

        # Scale the attention output before passing to the next block
        x = x * self.weight

        ffn_out = self.ffn(x)
        x = self.layer_norm2(ffn_out + x)  # Add & Norm

        return x, cached_k, cached_v

class GPT(nn.Module):
    def __init__(self, embed_dim=1000, num_heads=10, ff_dim=4000, num_layers=25, vocab_size=1001, dropout=0.1, max_seq_len=98_280):
        super().__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)
        self.layers = nn.ModuleList([GPTBlock(embed_dim, num_heads, ff_dim, dropout, layer_id=i) for i in range(num_layers)])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.decoder = GPTDecoder(embed_dim, vocab_size)

    def forward(self, x, use_cache=False):
        x = self.positional_encoding(x)

        cached_keys, cached_values = [], []
        for layer in self.layers:
            x, k, v = layer(x, use_cache=use_cache)
            cached_keys.append(k)
            cached_values.append(v)

        x = self.final_norm(x)
        logits = self.decoder(x)

        return logits, cached_keys, cached_values
