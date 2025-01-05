import matplotlib.pyplot as plt
import numpy as np
import json
import os
import itertools

val_dir = "val/"
metrics_file = "metrics.json"
# evaluation_dir = "evaluation/"
evaluation_dir = "evaluation_v2/"
dirs = sorted(os.listdir(evaluation_dir))

data_all = dict()
for dir in dirs:
    file_path = evaluation_dir + dir + "/" + val_dir + metrics_file

    with open(file_path, "r") as f:
        data = json.load(f)
        data_all[dir] = data

models = list(data_all.keys())
print(models)
metrics = list(data_all[models[0]].keys())
print(metrics)
values = np.array(
    [[model_data[metric] for metric in metrics] for model_data in data_all.values()]
)

# colors = "#474E93 #7E5CAD #72BAA9 #D5E7B5".split(" ")
colors = "#9467bd #17becf #d62728 #bcbd22 #474E93".split(" ")
x = np.arange(len(metrics))  # the label locations
width = 0.15  # width of bars

fig, ax = plt.subplots(figsize=(14, 6))
for i, (model, color) in enumerate(itertools.zip_longest(models, colors)):
    # print(x + i * width)
    # print(values[i])
    ax.bar(x + i * width, values[i], width, label=model, color=color)

# print(x)
# print(x + width / 2)

xticks = x + (width / 2) * (len(models) - 1)

# Add labels, title, and legend
ax.set_xlabel("Metrics")
ax.set_ylabel("Values")
ax.set_title("Comparison of YOLO Models Across Metrics")
# ax.set_xticks(x)                      # 1 bar
# ax.set_xticks(x + width / 2)          # 2 bars
# ax.set_xticks(x + width)              # 3 bars
# ax.set_xticks(x + width + width / 2)  # 4 bars
ax.set_xticks(xticks)                   # n bars
ax.set_xticklabels(map(lambda x: x.replace("metrics/", "").replace("(B)", ""), metrics))
ax.legend()

plt.tight_layout()
plt.show()
