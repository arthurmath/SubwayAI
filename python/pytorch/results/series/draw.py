import torch
import matplotlib.pyplot as plt

# 1. Load the data
data = torch.load("python/pytorch/results/series/history_20260410_225206.pth")
scores_history = data['scores_history']
rewards_history = data['rewards_history']

# 2. Extract values for plotting
iterations = [d['iteration'] for d in scores_history]
avg_scores = [d['avg_score'] for d in scores_history]

# 3. Draw a new graph
plt.figure(figsize=(10, 5))
plt.plot(iterations, avg_scores, label='Average Score')
plt.xlabel('Iteration')
plt.ylabel('Distance (m)')
plt.title('Reloaded Scores History')
plt.legend()
plt.grid(True)
plt.show()