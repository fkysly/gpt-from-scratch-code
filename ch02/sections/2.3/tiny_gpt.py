import torch

torch.manual_seed(1337)

text = "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
chars = sorted(set(text))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join([itos[i] for i in ids])


data = torch.tensor(encode(text))
block_size = 8
i = torch.randint(len(data) - block_size, (1,)).item()
x = data[i:i + block_size]
y = data[i + 1:i + block_size + 1]

print("起点：", i)
for t in range(block_size):
    context = x[:t + 1]
    target = y[t]
    print(decode(context.tolist()), "->", itos[target.item()])
