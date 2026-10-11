import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

text = "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join([itos[i] for i in ids])


data = torch.tensor(encode(text))
block_size = 8
batch_size = 4
learning_rate = 1e-2
max_steps = 1000


def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


class BigramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.table(idx)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss


model = BigramModel()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_steps):
    x, y = get_batch()
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(step, loss.item())

idx = torch.tensor([[stoi["床"]]])
for _ in range(30):
    logits, _ = model(idx)
    probs = F.softmax(logits[:, -1, :], dim=-1)
    idx_next = probs.argmax(dim=-1, keepdim=True)
    idx = torch.cat([idx, idx_next], dim=1)
print(decode(idx[0].tolist()))
