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

torch.manual_seed(1337)
prompt = "Once upon a time"
idx = torch.tensor([tokenizer.encode(prompt).ids], device=device)
out = model.generate(idx, max_new_tokens=200)
print(tokenizer.decode(out[0].tolist()))
