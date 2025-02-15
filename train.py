import torch
import torch.nn as nn
import torch.optim as optim
import os
from gpt import GPT
from embedding import embed_input
from data_loader import load_random_text

# Training Configuration
batch_size = 2
seq_len = 1024
embed_dim = 1000
vocab_size = 1001  # Matches embedding vocabulary
num_epochs = 10  # Number of training iterations
learning_rate = 1e-4

# Load vocabulary (replace with a real tokenizer's vocab)
vocab_dict = {"hello": 5, "world": 23, "GPT": 87, "training": 150}  # Example vocab

# Initialize model and optimizer
gpt = GPT(embed_dim=embed_dim, vocab_size=vocab_size)
optimizer = optim.Adam(gpt.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# Load pre-existing attention weights (if available)
def load_attention_weights(model):
    for i, layer in enumerate(model.layers):
        weight_path = f"attention_weights/layer_{i}.pth"
        if os.path.exists(weight_path):
            layer.attention.load_state_dict(torch.load(weight_path))
            print(f"✅ Loaded weights for Layer {i}")
        else:
            print(f"⚠️ No saved weights found for Layer {i}, using random initialization.")

load_attention_weights(gpt)

# Training Loop
for epoch in range(num_epochs):
    print(f"\n🚀 Epoch {epoch + 1}/{num_epochs}")

    # Load a random section of text and convert it to token indices
    token_indices = load_random_text("dataset.txt", seq_len, vocab_dict)

    # Ensure it has the correct batch size
    token_indices = token_indices.unsqueeze(0).expand(batch_size, -1)  # (batch_size, seq_len)

    # Convert to embeddings
    embedded_tokens = embed_input(token_indices)

    # Forward Pass
    logits, _, _ = gpt(embedded_tokens)

    # Reshape logits for CrossEntropyLoss (batch_size * seq_len, vocab_size)
    logits = logits.view(-1, vocab_size)
    target = token_indices.view(-1)  # Target should have the same shape

    # Compute Loss
    loss = loss_fn(logits, target)
    print(f"🎯 Loss: {loss.item()}")

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