import torch
import numpy as np

print("torch 版本：", torch.__version__)
print("numpy 版本：", np.__version__)
print(torch.tensor([1, 2, 3]) * 2)
print("CUDA 可用：", torch.cuda.is_available())
print("MPS 可用：", torch.backends.mps.is_available())
