import json
import re
import urllib.parse
import urllib.request

BASE = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/"
poems = []
for start in range(0, 5000, 1000):
    path = urllib.parse.quote(f"全唐诗/poet.tang.{start}.json")
    with urllib.request.urlopen(BASE + path) as f:
        items = json.load(f)
    for item in items:
        poem = "\n".join(item["paragraphs"])
        if re.fullmatch(r"[\u4e00-\u9fff，。\n]+", poem):
            poems.append(poem)

text = "\n\n".join(poems)
with open("poems.txt", "w", encoding="utf-8") as f:
    f.write(text)
print(len(poems), "首诗，共", len(text), "个字符")
print(text[:37])
