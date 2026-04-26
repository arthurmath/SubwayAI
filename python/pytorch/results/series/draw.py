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
    "python/pytorch/results/series/history_20260419_102224.pth"
]

for file in files:
    data = torch.load(file)
    all_scores_history.extend(data['scores_history'])

# 2. Extract values for plotting
iterations = [d['iteration'] for d in all_scores_history]
avg_scores = [d['avg_score'] for d in all_scores_history]
averages = moving_average(avg_scores)


# 3. Draw a new graph
plt.figure(figsize=(10, 5))
plt.plot(avg_scores, label='Scores')
plt.plot(averages, label='Moving Average')
plt.xlabel('Iteration')
plt.ylabel('Distance (m)')
plt.title('Scores and moving average')
plt.legend()
plt.grid(True)
plt.show()