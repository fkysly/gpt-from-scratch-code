text = "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
chars = sorted(set(text))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join([itos[i] for i in ids])


print("词表大小：", len(chars))
print(decode([12, 14]))
print(decode(encode(text)))
