text = "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
chars = sorted(set(text))

for i, ch in enumerate(chars):
    print(i, ch, end="  ")
print()
