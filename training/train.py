
import torch
import torch.nn as nn
import torch.optim as optim
from model.transformer_model import MiniTransformer
from training.utils import encode_text, decode_tokens
import model.config as config
import random

# Prepare data
with open('data/tiny_shakespeare.txt', 'r') as f:
    text = f.read()

vocab = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}

data = encode_text(text, stoi)
seq_len = 50

def get_batch():
    start = random.randint(0, len(data) - seq_len - 1)
    x = data[start:start+seq_len]
    y = data[start+1:start+seq_len+1]
    return x.unsqueeze(0), y.unsqueeze(0)

model = MiniTransformer(len(vocab), config.embedding_dim, config.hidden_dim,
                        config.num_heads, config.num_layers, config.max_seq_len)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

for epoch in range(config.epochs):
    model.train()
    x, y = get_batch()
    logits = model(x)
    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 2 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        model.eval()
        with torch.no_grad():
            sample = x[0][:10].tolist()
            for _ in range(100):
                input_tensor = torch.tensor(sample[-seq_len:], dtype=torch.long).unsqueeze(0)
                output = model(input_tensor)
                next_token = torch.argmax(output[0, -1]).item()
                sample.append(next_token)
            with open("output/generated_text.txt", "w") as f:
                f.write(decode_tokens(sample, itos))
