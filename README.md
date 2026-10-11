# 从零写一个迷你 GPT：配套代码

每章一个目录，里面是**这一章结束时**的完整代码。每一章都在上一章的代码上继续写，不清空。
所有代码都由书稿构建脚本自动生成，和书中内容逐字一致。

## 环境

```bash
uv sync          # 按 pyproject.toml 安装依赖（torch、numpy、tokenizers、matplotlib）
cd ch05
uv run --project .. tiny_gpt.py
```

根目录的 `.python-version` 把 Python 固定为 3.13（PyTorch 还没有更新版本 Python 的安装包）。
第 0–9 章在任何电脑的 CPU 上都能运行；第 10–12 章需要一块 NVIDIA GPU（自己的显卡或租用的云 GPU）。
数据文件（poems.txt、TinyStories、*.bin、ckpt.pt 等）不在仓库里，由各章的脚本下载或生成。

## 可选：AMD GPU（ROCm）

第 10–12 章的代码不用修改也能在 AMD GPU 上运行（ROCm 版 PyTorch 下 `torch.cuda.is_available()` 为 True，`device="cuda"` 照常可用）。以 AMD Developer Cloud 的 PyTorch 镜像为例：先 `docker exec -it rocm bash` 进入容器，`pip install tokenizers`，然后用 `python prepare.py`、`python train.py` 等代替 `uv run`。**不要在那里运行 `uv sync`**：`uv.lock` 锁定的是 PyPI 上的（NVIDIA 版）torch，会替换掉镜像里的 ROCm 版本，GPU 很可能就用不上了。

## 目录

| 目录 | 章节 | 文件 |
|---|---|---|
| [`ch00/`](ch00/) | 第 0 章　准备环境 | `check_env.py` |
| [`ch01/`](ch01/) | 第 1 章　从一首诗开始：把文字变成数字 | `tiny_gpt.py` |
| [`ch02/`](ch02/) | 第 2 章　带上前文：张量与批次 | `tiny_gpt.py` |
| [`ch03/`](ch03/) | 第 3 章　第一个会猜字的模型：打分表、softmax 与交叉熵 | `tiny_gpt.py` |
| [`ch04/`](ch04/) | 第 4 章　让它学起来：训练循环 | `tiny_gpt.py` |
| [`ch05/`](ch05/) | 第 5 章　让它写诗：一个字一个字地生成 | `tiny_gpt.py` |
| [`ch06/`](ch06/) | 第 6 章　让每个字看见前文：自注意力 | `tiny_gpt.py` |
| [`ch07/`](ch07/) | 第 7 章　拼成 Transformer：多头、前馈、残差与 LayerNorm | `tiny_gpt.py`、`get_poems.py` |
| [`ch08/`](ch08/) | 第 8 章　完整的 TinyGPT：验证集、过拟合与 dropout | `tiny_gpt.py`、`get_poems.py` |
| [`ch09/`](ch09/) | 第 9 章　更大的世界：TinyStories 与 BPE 分词 | `tiny_gpt.py`、`prepare.py` |
| [`ch10/`](ch10/) | 第 10 章　换上 GPU：更大的模型、检查点与损失曲线 | `prepare.py`、`model.py`、`train.py`、`plot.py`、`sample.py` |
| [`ch11/`](ch11/) | 第 11 章　控制生成：温度与 top-k | `prepare.py`、`model.py`、`train.py`、`plot.py`、`sample.py` |
| [`ch12/`](ch12/) | 第 12 章　教它对话：指令微调与聊天 | `prepare.py`、`model.py`、`train.py`、`plot.py`、`sample.py`、`sft_data.py`、`sft.py`、`chat.py` |

## License

MIT. See [LICENSE](LICENSE).

This repository is the companion code for the tutorial book. The book text is separate; this repo contains only the runnable code for each chapter.
