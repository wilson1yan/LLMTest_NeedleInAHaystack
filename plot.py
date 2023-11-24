import argparse
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_folder', type=str, default='results')
parser.add_argument('--path_substr', type=str, required=True)
args = parser.parse_args()

data_folder = Path(args.data_folder)
data_files = list(data_folder.glob(f"{args.path_substr}*.json"))

data = []
for data_file in data_files:
    data.extend(json.load(data_file.open('r')))
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
        grid[i, j] = np.mean(mapping.get((depth_vals[i], context_vals[j]), 0))
print(grid)
