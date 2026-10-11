import json
import random

random.seed(1337)
EOT = "<|endoftext|>"
with open("TinyStoriesV2-GPT4-valid.txt", encoding="utf-8") as f:
    stories = [s.strip() for s in f.read().split(EOT) if s.strip()]

NAMES = ["Lily", "Tom", "Ben", "Sue", "Max", "Mia", "Tim", "Sam", "Anna",
         "Lucy", "Jack", "Sara", "Timmy", "Amy"]
THINGS = ["dog", "cat", "ball", "tree", "park", "cake", "bird", "car", "boat",
          "toy", "flower", "fish", "rabbit", "bear"]
CHATS = [
    ("Hi!", "Hi! I am TinyGPT. I like to tell stories. What story do you want?"),
    ("Hello", "Hello! Do you want me to tell you a story?"),
    ("Who are you?", "I am TinyGPT, a very small language model. I tell stories."),
    ("What can you do?", "I can tell short stories. Ask me for a story about a cat!"),
    ("Thank you!", "You are welcome! I hope you liked the story."),
    ("Bye", "Bye! Come back for another story soon."),
]

examples = []
for story in stories:
    clean = story
    for mark in '.,!?"':
        clean = clean.replace(mark, " ")
    words = set(clean.split())
    names = [n for n in NAMES if n in words]
    things = [t for t in THINGS if t in words]
    r = random.random()
    if names and r < 0.4:
        prompt = f"Write a story about {random.choice(names)}."
        examples.append({"prompt": prompt, "response": story})
    elif things and r < 0.8:
        prompt = f"Tell me a story about a {random.choice(things)}."
        examples.append({"prompt": prompt, "response": story})
    elif ". " in story:
        first, rest = story.split(". ", 1)
        prompt = f"Continue this story: {first}."
        examples.append({"prompt": prompt, "response": rest})
for _ in range(100):
    for prompt, response in CHATS:
        examples.append({"prompt": prompt, "response": response})
random.shuffle(examples)

with open("sft.jsonl", "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")
print(len(examples), "条对话")
for ex in examples[:3]:
    print(ex["prompt"], "->", ex["response"][:60], "...")
