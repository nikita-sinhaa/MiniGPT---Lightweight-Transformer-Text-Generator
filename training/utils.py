
import torch

def encode_text(text, stoi):
    return torch.tensor([stoi[c] for c in text if c in stoi], dtype=torch.long)

def decode_tokens(tokens, itos):
    return ''.join([itos[i] for i in tokens])
