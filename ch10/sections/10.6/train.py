import math
import time
from dataclasses import asdict

import numpy as np
import torch

from model import GPTConfig, TinyGPT

batch_size = 64
max_steps = 5000
eval_interval = 250
eval_iters = 50
learning_rate = 1e-3
min_lr = 1e-4
warmup_steps = 200
grad_clip = 1.0
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
use_bf16 = device == "cuda"

torch.manual_seed(1337)
config = GPTConfig()
train_data = np.memmap("train.bin", dtype=np.uint16, mode="r")
val_data = np.memmap("val.bin", dtype=np.uint16, mode="r")


def get_batch(split):
    data = train_data if split == "train" else val_data
    T = config.block_size
    ix = torch.randint(len(data) - T, (batch_size,))
    x = np.stack([data[i:i + T] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + T] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


model = TinyGPT(config).to(device)
print(sum(p.numel() for p in model.parameters()) / 1e6, "M 个参数，设备：", device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(step):
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (learning_rate - min_lr)


best_val = float("inf")
log = open("log.csv", "w")
log.write("step,train,val\n")
t0 = time.time()
for step in range(max_steps):
    lr = get_lr(step)
    for group in optimizer.param_groups:
        group["lr"] = lr
    if step % eval_interval == 0 or step == max_steps - 1:
        losses = estimate_loss()
        tr, va = losses["train"], losses["val"]
        elapsed = time.time() - t0
        print(f"step {step:5d} | train {tr:.3f} | val {va:.3f} | "
              f"lr {lr:.2e} | {elapsed:.0f}s")
        log.write(f"{step},{tr:.4f},{va:.4f}\n")
        log.flush()
        if va < best_val:
            best_val = va
            ckpt = {"model": model.state_dict(), "config": asdict(config),
                    "step": step, "val_loss": va}
            torch.save(ckpt, "ckpt.pt")
    x, y = get_batch("train")
    with torch.autocast(device, dtype=torch.bfloat16, enabled=use_bf16):
        logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
