import os
import urllib.request

URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
FILES = ["TinyStoriesV2-GPT4-valid.txt"]
EOT = "<|endoftext|>"


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
print(stories[1])
