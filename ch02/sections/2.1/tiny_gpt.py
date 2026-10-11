text = "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
chars = sorted(set(text))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join([itos[i] for i in ids])


data = encode(text)
block_size = 8
x = data[0:block_size]
y = data[1:block_size + 1]

for t in range(block_size):
    context = x[:t + 1]
    target = y[t]
    print(decode(context), "->", itos[target])
