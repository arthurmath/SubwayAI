
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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





def save_plots(scores_history, rewards_history):
    """
    scores_history: list of dict {'iteration': int, 'avg_score': float, 'best_score': float}
    rewards_history: list of dict {'iteration': int, 'avg_reward': float, 'best_reward': float}
    """
    os.makedirs("results/plots", exist_ok=True)
    
    if not scores_history or not rewards_history:
        print("No data to plot.")
        return

    iterations_s = [d['iteration'] for d in scores_history]
    avg_scores = [d['avg_score'] for d in scores_history]
    best_scores = [d['best_score'] for d in scores_history]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations_s, avg_scores, label='Score Moyen (m)')
    plt.plot(iterations_s, best_scores, label='Meilleur Score (m)')
    plt.xlabel('Itération')
    plt.ylabel('Distance (m)')
    plt.title('Score en fonction de l\'itération')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/scores.png')
    plt.close()

    iterations_r = [d['iteration'] for d in rewards_history]
    avg_rewards = [d['avg_reward'] for d in rewards_history]
    best_rewards = [d['best_reward'] for d in rewards_history]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations_r, avg_rewards, label='Reward Moyenne')
    plt.plot(iterations_r, best_rewards, label='Meilleure Reward')
    plt.xlabel('Itération')
    plt.ylabel('Reward')
    plt.title('Reward en fonction de l\'itération')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/rewards.png')
    plt.close()
    print("Plots saved in results/plots/")






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

