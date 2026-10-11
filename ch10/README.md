# 第 10 章　换上 GPU：更大的模型、检查点与损失曲线

本目录是这一章结束时的完整代码，在本目录下运行：

```bash
uv run prepare.py     # 需要 GPU 机器；下载 2.2 GB 训练集
uv run train.py       # GB10 上约 21 分钟
uv run plot.py
uv run sample.py
```

`sections/` 里是每一节结束时的代码：

- `sections/10.1/`：`tiny_gpt.py`、`prepare.py`
- `sections/10.2/`：`prepare.py`、`model.py`
- `sections/10.3/`：`prepare.py`、`model.py`、`train.py`
- `sections/10.4/`：`prepare.py`、`model.py`、`train.py`
- `sections/10.5/`：`prepare.py`、`model.py`、`train.py`
- `sections/10.6/`：`prepare.py`、`model.py`、`train.py`
- `sections/10.7/`：`prepare.py`、`model.py`、`train.py`
- `sections/10.8/`：`prepare.py`、`model.py`、`train.py`、`plot.py`、`sample.py`

