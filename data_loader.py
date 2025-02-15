import torch
import random
import os

# Define a simple tokenizer (replace with a real tokenizer if needed)
def basic_tokenizer(text, vocab_dict):
    words = text.split()
    tokens = [vocab_dict.get(word, 0) for word in words]  # Convert words to token indices
    return torch.tensor(tokens, dtype=torch.long)

# Function to load random text data from a file
def load_random_text(file_path, seq_len, vocab_dict):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file {file_path} not found!")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) == 0:
        raise ValueError("Dataset file is empty!")

    # Randomly pick a line and take a chunk of text
    random_line = random.choice(lines).strip()  # Select a random line
    words = random_line.split()

    # Ensure we get at least `seq_len` words
    if len(words) > seq_len:
        start_index = random.randint(0, len(words) - seq_len)
        words = words[start_index:start_index + seq_len]  # Select a chunk

    text_chunk = " ".join(words)  # Join words back into a string

    # Convert to token indices
    token_indices = basic_tokenizer(text_chunk, vocab_dict)
    
    return token_indices