import json
import random

import torch
from tokenizers import Tokenizer

from model import GPTConfig, TinyGPT

batch_size = 32
max_steps = 1000
eval_interval = 100
learning_rate = 1e-4
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
use_bf16 = device == "cuda"

torch.manual_seed(1337)
random.seed(1337)
ckpt = torch.load("ckpt.pt", map_location=device)
config = GPTConfig(**ckpt["config"])
model = TinyGPT(config).to(device)
model.load_state_dict(ckpt["model"])
tokenizer = Tokenizer.from_file("tokenizer.json")
eot = tokenizer.token_to_id("<|endoftext|>")


def encode_example(ex):
    p = tokenizer.encode(f"User: {ex['prompt']}\nAssistant:").ids
    r = tokenizer.encode(" " + ex["response"]).ids + [eot]
    ids = p + r
    x = ids[:-1]
    y = [-100] * (len(p) - 1) + ids[len(p):]
    return x, y


with open("sft.jsonl", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
examples = [encode_example(ex) for ex in data]
examples = [ex for ex in examples if len(ex[0]) <= config.block_size]
val_examples, train_examples = examples[:200], examples[200:]
print(len(train_examples), "条训练样本，", len(val_examples), "条验证样本")

x, y = train_examples[0]
print("输入：", tokenizer.decode(x))
print("算损失的部分：", tokenizer.decode([t for t in y if t != -100]))


def get_batch(split):
    pool = train_examples if split == "train" else val_examples
    batch = random.sample(pool, batch_size)
    T = max(len(x) for x, _ in batch)
    X = torch.full((batch_size, T), eot)
    Y = torch.full((batch_size, T), -100)
    for i, (x, y) in enumerate(batch):
        X[i, :len(x)] = torch.tensor(x)
        Y[i, :len(y)] = torch.tensor(y)
    return X.to(device), Y.to(device)
