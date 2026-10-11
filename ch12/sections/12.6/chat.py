import torch
from tokenizers import Tokenizer

from model import GPTConfig, TinyGPT

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
ckpt = torch.load("sft.pt", map_location=device)
model = TinyGPT(GPTConfig(**ckpt["config"])).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
tokenizer = Tokenizer.from_file("tokenizer.json")
eot = tokenizer.token_to_id("<|endoftext|>")

torch.manual_seed(1337)
print("和 TinyGPT 聊天，输入 quit 退出。")
while True:
    user = input("你：")
    if user.strip() == "quit":
        break
    prompt = f"User: {user}\nAssistant:"
    idx = torch.tensor([tokenizer.encode(prompt).ids], device=device)
    out = model.generate(idx, max_new_tokens=300, temperature=0.8, top_k=20,
                         stop_token=eot)
    reply = tokenizer.decode(out[0, idx.shape[1]:].tolist()).strip()
    print("TinyGPT：", reply)
