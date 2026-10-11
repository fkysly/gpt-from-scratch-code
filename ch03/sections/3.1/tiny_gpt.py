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


def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


table = nn.Embedding(vocab_size, vocab_size)

x, y = get_batch()
logits = table(x)
print(x.shape)
print(logits.shape)
