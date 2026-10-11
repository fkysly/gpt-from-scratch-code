import torch
from tokenizers import Tokenizer

from model import GPTConfig, TinyGPT

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
ckpt = torch.load("ckpt.pt", map_location=device)
model = TinyGPT(GPTConfig(**ckpt["config"])).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
tokenizer = Tokenizer.from_file("tokenizer.json")
print("第", ckpt["step"], "步的检查点，验证损失", round(ckpt["val_loss"], 3))
eot = tokenizer.token_to_id("<|endoftext|>")


def write(prompt, temperature=1.0, top_k=None, seed=1337):
    torch.manual_seed(seed)
    idx = torch.tensor([tokenizer.encode(prompt).ids], device=device)
    out = model.generate(idx, max_new_tokens=300, temperature=temperature,
                         top_k=top_k, stop_token=eot)
    return tokenizer.decode(out[0].tolist())


for temperature, top_k in [(0.3, None), (1.0, None), (1.5, None), (1.5, 20)]:
    print(f"=== temperature={temperature}, top_k={top_k} ===")
    print(write("Once upon a time", temperature, top_k))
    print()
