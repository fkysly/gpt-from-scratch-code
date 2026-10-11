import os
import urllib.request

import numpy as np

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
FILES = ["TinyStoriesV2-GPT4-train.txt", "TinyStoriesV2-GPT4-valid.txt"]
EOT = "<|endoftext|>"
vocab_size = 4096


def download(name):
    if not os.path.exists(name):
        print("下载", name, "……")
        urllib.request.urlretrieve(URL + name, name + ".part")
        os.rename(name + ".part", name)
    return name


def read_stories(name):
    stories, lines = [], []
    with open(download(name), encoding="utf-8") as f:
        for line in f:
            if line.strip() == EOT:
                stories.append("".join(lines).strip())
                lines = []
            else:
                lines.append(line)
    stories.append("".join(lines).strip())
    return [s for s in stories if s]


stories = []
for name in FILES:
    stories += read_stories(name)
print(len(stories), "个故事")
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    special_tokens=[EOT],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=False,
)
tokenizer.train_from_iterator(stories[:200_000], trainer)
tokenizer.save("tokenizer.json")

enc = tokenizer.encode("Once upon a time, there was a little girl named Lily.")
print(enc.tokens)
print(enc.ids)
print(tokenizer.decode(enc.ids))
eot = tokenizer.token_to_id(EOT)
chunks = []
for i in range(0, len(stories), 10_000):
    for e in tokenizer.encode_batch(stories[i:i + 10_000]):
        chunks.append(np.array(e.ids + [eot], dtype=np.uint16))
ids = np.concatenate(chunks)

n = int(0.9 * len(ids))
ids[:n].tofile("train.bin")
ids[n:].tofile("val.bin")
print("编号总数：", len(ids), "训练：", n, "验证：", len(ids) - n)
print("平均每个编号对应", round(sum(len(s) + 1 for s in stories) / len(ids), 2), "个字符")
