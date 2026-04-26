
import os
import glob
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights_dir = "python/pytorch/results/weights"
os.makedirs(weights_dir, exist_ok=True)


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def extend(self, other):
        self.actions.extend(other.actions)
        self.states.extend(other.states)
        self.logprobs.extend(other.logprobs)
        self.rewards.extend(other.rewards)
        self.state_values.extend(other.state_values)
        self.is_terminals.extend(other.is_terminals)





def format(buffer):
    old_states = torch.squeeze(torch.stack(buffer.states, dim=0)).detach().to(device)
    old_actions = torch.squeeze(torch.stack(buffer.actions, dim=0)).detach().to(device)
    old_logprobs = torch.squeeze(torch.stack(buffer.logprobs, dim=0)).detach().to(device)
    old_state_values = torch.squeeze(torch.stack(buffer.state_values, dim=0)).detach().to(device)
    
    # Ensure correct dimensions if batch size is 1
    if len(old_states.shape) == 1:
        old_states = old_states.unsqueeze(0)
    if len(old_actions.shape) == 0:
        old_actions = old_actions.unsqueeze(0)
    if len(old_logprobs.shape) == 0:
        old_logprobs = old_logprobs.unsqueeze(0)
    if len(old_state_values.shape) == 0:
        old_state_values = old_state_values.unsqueeze(0)

    return old_states, old_actions, old_logprobs, old_state_values






def extract_state(game_state):
    player = game_state.get('player', {})
    obstacles = game_state.get('obstacles', [])
    coins = game_state.get('coins', [])
    speed = game_state.get('speed', 10.0)
    
    # 1. Player state
    state = [
        player.get('lane', 1) - 1.0, # Values: -1.0, 0.0, 1.0
        player.get('y', 0) / 3.0,
        1.0 if player.get('rolling', False) else 0.0, # sliding (bool 0 or 1)
        speed / 10.0
    ]
    
    # 2. Obstacles: Track closest per lane
    obs_by_lane = {}
    for lane in range(3):
        # find closest obstacle in this lane that is ahead (z < 0)
        obs_in_lane = [o for o in obstacles if o.get('lane') == lane and o.get('z', 0) < 0]
        obs_in_lane.sort(key=lambda o: o.get('z', 0), reverse=True) # closest is biggest z (least negative)
        
        if obs_in_lane:
            obs = obs_in_lane[0]
            obs_z = obs.get('z', 0)
            z_norm = abs(obs_z) / 50.0 # distance normalized
            t = obs.get('type', 'low')
            if t == 'low': otype = 0.0
            elif t == 'high': otype = 0.5
            else: otype = 1.0 # train
            state.extend([z_norm, otype])
            obs_by_lane[lane] = obs_z
        else:
            state.extend([1.0, -1.0]) # Z = 1.0 (very far), Type = -1.0 (Nothing)
            obs_by_lane[lane] = -500.0 # No obstacle, very far
            
    # 3. Coins: Track next per lane (only if before obstacle)
    for lane in range(3):
        coins_in_lane = [c for c in coins if c.get('lane') == lane and c.get('z', 0) < 0]
        coins_in_lane.sort(key=lambda c: c.get('z', 0), reverse=True)
        
        obs_z = obs_by_lane[lane]
        # Only consider coins that are before the next obstacle in this lane
        coins_before_obs = [c for c in coins_in_lane if c.get('z', 0) > obs_z]
        
        if coins_before_obs:
            first_coin = coins_before_obs[0]
            z_norm = abs(first_coin.get('z', 0)) / 50.0
            count = len(coins_before_obs)
            state.extend([z_norm, float(count)])
        else:
            state.extend([1.0, 0.0]) # No coin before obstacle
            
    return state




def save_weights(policy, score):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{weights_dir}/score_{int(score)}_{timestamp}.pth"
    torch.save(policy.state_dict(), filename)
    print(f"Saved weights to {filename}")




def load_best(policy, policy_old):
    files = glob.glob(f"{weights_dir}/score_*.pth")
    if not files:
        print("No weights found to load.")
        return False

    best_file = None
    best_score = -1
    for f in files:
        try:
            base = os.path.basename(f)
            score_str = base.split("_")[2].replace(".pth", "")
            score = float(score_str)
            if score > best_score:
                best_score = score
                best_file = f
        except Exception:
            pass

    if best_file:
        print(f"Loading best weights: {best_file}")
        state_dict = torch.load(best_file, map_location=device, weights_only=True)
        policy.load_state_dict(state_dict)
        policy_old.load_state_dict(state_dict)
        return True
    return False




def moving_average(values):
    len_window = len(values) // 8
    moyenne_mobile = []
    for i in range(len(values)):
        if i < len_window:
            start_index = 0
        else:
            start_index = i - len_window + 1
        window = values[start_index: i + 1]
        moyenne_mobile.append(sum(window) / len(window))
    return moyenne_mobile




def save_plots(scores_history, rewards_history):
    """
    scores_history: list of dict {'iteration': int, 'avg_score': float, 'best_score': float}
    rewards_history: list of dict {'iteration': int, 'avg_reward': float, 'best_reward': float}
    """
    matplotlib.use('Agg')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not scores_history or not rewards_history:
        print("No data to plot.")
        return

    # Save history lists to a file
    series_path = f"python/pytorch/results/series/history_{timestamp}.pth"
    torch.save({
        'scores_history': scores_history,
        'rewards_history': rewards_history
    }, series_path)
    print(f"History lists saved to {series_path}")

    iterations_s = [d['iteration'] for d in scores_history]
    avg_scores   = [d['avg_score']  for d in scores_history]
    best_scores  = [d['best_score'] for d in scores_history]

    iterations_r = [d['iteration']   for d in rewards_history]
    avg_rewards  = [d['avg_reward']  for d in rewards_history]
    raw_best_rewards = [d['best_reward'] for d in rewards_history]
    best_rewards = list(np.maximum.accumulate(raw_best_rewards))

    def _make_figure(iterations, best_values, raw_values, best_label, raw_label, y_label, title_best, title_raw, filepath):
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 10))

        ax_top.plot(iterations, best_values, color='orange', label=best_label)
        ax_top.set_xlabel('Itération')
        ax_top.set_ylabel(y_label)
        ax_top.set_title(title_best)
        ax_top.legend()
        ax_top.grid(True)

        ax_bot.plot(iterations, raw_values, color='blue', alpha=0.5, label=raw_label)
        ma = moving_average(raw_values)
        ma_iterations = iterations[len(iterations) - len(ma):]
        ax_bot.plot(ma_iterations, ma, color='black', linewidth=2, label='Moyenne mobile')
        ax_bot.set_xlabel('Itération')
        ax_bot.set_ylabel(y_label)
        ax_bot.set_title(title_raw)
        ax_bot.legend()
        ax_bot.grid(True)

        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

    _make_figure(
        iterations_s, best_scores, avg_scores,
        best_label='Meilleur Score (m)', raw_label='Score Moyen (m)',
        y_label='Distance (m)',
        title_best='Meilleur Score en fonction de l\'itération',
        title_raw='Score en fonction de l\'itération',
        filepath=f"python/pytorch/results/plots/scores_{timestamp}.png",
    )

    _make_figure(
        iterations_r, best_rewards, avg_rewards,
        best_label='Meilleure Reward', raw_label='Reward Moyenne',
        y_label='Reward',
        title_best='Meilleure Reward en fonction de l\'itération',
        title_raw='Reward en fonction de l\'itération',
        filepath=f"python/pytorch/results/plots/rewards_{timestamp}.png",
    )

    print("Plots saved in python/pytorch/results/plots/")



if __name__ == "__main__":
    # Test with synthetic data
    import random

    current_score = 7.0
    num_iterations = 1500
    # Generate synthetic data with an upward trend
    scores_history = [current_score := current_score + random.uniform(-20.0, 25) for _ in range(num_iterations)]
    averages = moving_average(scores_history)

    plt.figure(figsize=(10, 5))
    plt.plot(scores_history, label='Scores')
    plt.plot(averages, label='Averages')
    plt.xlabel('Itération')
    plt.ylabel('Score')
    plt.title('Scores et moyennes mobiles (synthétiques)')
    plt.legend()
    plt.tight_layout()
    plt.show()




# import numpy as np

# async def play_random(websocket):
#     print("New connection from game client!")
#     try:
#         async for message in websocket:
#             # 1. Receive game state from JS
#             game_state = json.loads(message)

#             # Manually set custom probability array for actions.
#             probs = np.array([0.1, 0.1, 0.1, 0.1, 0.6], dtype=np.float32)
#             action_idx = np.random.choice(len(ACTIONS), p=probs)
#             response = {"action": ACTIONS[action_idx]}
#             await websocket.send(json.dumps(response))
#     except websockets.exceptions.ConnectionClosed:
#         print("Client disconnected.")





# def moving_average(values, window=20):
#     if len(values) < window:
#         window = max(1, len(values))
#     kernel = np.ones(window) / window
#     return np.convolve(values, kernel, mode='valid')