import os
import urllib.request

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
FILES = ["TinyStoriesV2-GPT4-valid.txt"]
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
