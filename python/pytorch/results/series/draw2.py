# Draw unique plot from different history files

import torch
import matplotlib.pyplot as plt
import sys
import os

# Add the directory containing utils.py to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from utils import moving_average

# 1. Load the data
all_scores_history = []

files = [
    "python/pytorch/results/series/history_20260411_085311.pth",
    "python/pytorch/results/series/history_20260412_121748.pth",
    "python/pytorch/results/series/history_20260419_102224.pth",
    "python/pytorch/results/series/history_20260419_220344.pth",
]

for file in files:
    data = torch.load(file)
    all_scores_history.extend(data['scores_history'])

# 2. Extract values for plotting
iterations = [d['iteration'] for d in all_scores_history]
avg_scores = [d['avg_score'] for d in all_scores_history]
averages = moving_average(avg_scores)



all_scores_history2 = []

files2 = [
    "python/pytorch/results/series/history_20260427_094736.pth",
    "python/pytorch/results/series/history_20260502_103808.pth",
]

for file in files2:
    data = torch.load(file)
    all_scores_history2.extend(data['scores_history'])


# 2. Extract values for plotting
iterations2 = [d['iteration'] for d in all_scores_history2]
avg_scores2 = [d['avg_score'] for d in all_scores_history2]
averages2 = moving_average(avg_scores2)


# 3. Draw a new graph
plt.figure(figsize=(10, 5))
plt.plot(averages2, label='Scores 2')
# plt.plot(averages, label='Scores 1')
plt.xlabel('Iteration')
plt.ylabel('Distance (m)')
plt.title('Moving averages')
plt.legend()
plt.grid(True)
plt.show()