import csv

import matplotlib.pyplot as plt

steps, train, val = [], [], []
with open("log.csv") as f:
    for row in csv.DictReader(f):
        steps.append(int(row["step"]))
        train.append(float(row["train"]))
        val.append(float(row["val"]))

plt.plot(steps, train, label="train")
plt.plot(steps, val, label="val", linestyle="--")
plt.xlabel("step")
plt.ylabel("loss")
plt.ylim(min(train + val) - 0.2, 4)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("loss.png", dpi=150)
print("已保存 loss.png")
