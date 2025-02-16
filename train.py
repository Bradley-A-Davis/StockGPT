import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import numpy as np
from GPT import GPT
from embedding import embed_input

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load percent change values from file
input_file = "Post Data/AAPL2Week.txt"
data = pd.read_csv(input_file, header=None, names=["percent_change"])
values = data["percent_change"].values

# Create a vocabulary dictionary mapping each unique percent change to an index
unique_values = np.unique(values)
vocab_dict = {value: idx for idx, value in enumerate(unique_values)}

# Training Configuration
batch_size = 1
seq_len = 1024
embed_dim = 1000  # Ensure it is a multiple of 8 for memory-efficient attention
vocab_size = len(vocab_dict)  # Matches embedding vocabulary
num_epochs = 1  # Number of training iterations
learning_rate = 1e-4

# Initialize model and optimizer
gpt = GPT(embed_dim=embed_dim, vocab_size=vocab_size).to(device)
optimizer = optim.Adam(gpt.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# Load pre-existing attention weights (if available)
def load_attention_weights(model):
    for i, layer in enumerate(model.layers):
        weight_path = f"attention_weights/layer_{i}.pth"
        if os.path.exists(weight_path):
            layer.attention.load_state_dict(torch.load(weight_path, map_location=device))
            print(f"✅ Loaded weights for Layer {i}")
        else:
            print(f"⚠️ No saved weights found for Layer {i}, using random initialization.")

load_attention_weights(gpt)

# Convert percent change values into token indices
def get_tokenized_input(data, seq_len, vocab_dict):
    indices = [vocab_dict[value] for value in data if value in vocab_dict]
    if len(indices) < seq_len:
        indices = indices + [0] * (seq_len - len(indices))  # Pad if needed
    return torch.tensor(indices[:seq_len], dtype=torch.long, device=device)

# Training Loop
for epoch in range(num_epochs):
    print(f"\n🚀 Epoch {epoch + 1}/{num_epochs}")

    # Prepare token indices for training
    token_indices = get_tokenized_input(values, seq_len, vocab_dict).to(device)

    # Ensure correct batch size
    token_indices = token_indices.unsqueeze(0).expand(batch_size, -1).to(device)  # (batch_size, seq_len)

    # Convert to embeddings
    embedded_tokens = embed_input(token_indices).to(device)

    # Forward Pass
    logits, _, _ = gpt(embedded_tokens)

    # Reshape logits for CrossEntropyLoss (batch_size * seq_len, vocab_size)
    logits = logits.view(-1, vocab_size)
    target = token_indices.view(-1)  # Target should have the same shape

    # Compute Loss
    loss = loss_fn(logits, target)
    print(f"🎯 Loss: {loss.item()}")

    # Print expected vs. actual output
    predicted_indices = torch.argmax(logits, dim=-1)
    expected_values = [list(vocab_dict.keys())[list(vocab_dict.values()).index(idx)] for idx in target.cpu().numpy()]
    actual_values = [list(vocab_dict.keys())[list(vocab_dict.values()).index(idx)] for idx in predicted_indices.cpu().numpy()]
    print(f"🟢 Expected: {expected_values[:10]}")  # Print first 10 values
    print(f"🔴 Actual:   {actual_values[:10]}")

    # Backward Pass (Backpropagation)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save Updated Weights
    def save_attention_weights(model):
        os.makedirs("attention_weights", exist_ok=True)
        for i, layer in enumerate(model.layers):
            weight_path = f"attention_weights/layer_{i}.pth"
            torch.save(layer.attention.state_dict(), weight_path)
            print(f"💾 Saved updated weights for Layer {i}")

    save_attention_weights(gpt)

print("\n✅ Training Complete! Model weights updated.")
