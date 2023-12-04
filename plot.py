import argparse
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_file', type=str, required=True)
args = parser.parse_args()

data = json.load(open(args.data_file, 'r'))
print(len(data))

depth_vals, context_vals = [], []
mapping = {}
for d in data:
    depth, context = d['depth_percent'], d['context_length']
    depth_vals.append(depth)
    context_vals.append(context)
    key = (depth, context)
    if key not in mapping:
        mapping[key] = []
    mapping[key].append(d['score'])
depth_vals = sorted(list(set(depth_vals)))
context_vals = sorted(list(set(context_vals)))

print(context_vals)
grid = np.zeros((len(depth_vals), len(context_vals)), dtype=np.float32)
for i in range(len(depth_vals)):
    for j in range(len(context_vals)):
        grid[i, j] = np.mean(mapping.get((depth_vals[i], context_vals[j]), [0]))

grid = grid / 10
cmap = plt.cm.get_cmap('viridis')
fig, ax = plt.subplots(figsize=(10, 6))
cax = ax.imshow(grid, cmap=cmap, aspect='auto')

cbar = fig.colorbar(cax)

x_ticks = np.arange(grid.shape[1])
ax.set_xticks(x_ticks)
ax.set_xticks(x_ticks - 0.5, minor=True)
xlabels = []
for c in context_vals:
    xlabels.append(f"{int(c / 1000)}k")
ax.set_xticklabels(xlabels, fontsize=8)

y_ticks = np.arange(grid.shape[0])
ax.set_yticks(y_ticks)
ax.set_yticks(y_ticks - 0.5, minor=True)
ylabels = []
for d in depth_vals:
    ylabels.append(f"{d*100:.1f}%")
ax.set_yticklabels(ylabels, fontsize=8)

ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5)

plt.title(Path(args.data_file).stem)
plt.xlabel('Context Length')
plt.ylabel('Document Depth')

plt.savefig('plot.png')

#for row in grid:
#    print(row.tolist())
