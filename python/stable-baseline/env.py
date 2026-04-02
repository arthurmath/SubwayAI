import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import asyncio
import websockets
import threading
import queue
import logging

logging.getLogger('websockets').setLevel(logging.ERROR)

ACTIONS = ['L', 'R', 'J', 'S', None]

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
            
    return np.array(state, dtype=np.float32)

class SubwaySurfersEnv(gym.Env):
    def __init__(self, port=8765):
        super(SubwaySurfersEnv, self).__init__()
        
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        
        self.port = port
        self.action_queue = queue.Queue()
        self.state_queue = queue.Queue()
        
        self.last_score = 0
        self.last_coins = 0
        self.session_best_score = 0
        self.iteration_count = 0
        self.train_count = 0
        
        # Start the websocket server in a separate thread
        self.loop = asyncio.new_event_loop()
        self.server_thread = threading.Thread(target=self._start_server, daemon=True)
        self.server_thread.start()
        
        # Wait for the first connection
        print(f"Waiting for game client to connect on ws://127.0.0.1:{self.port}...")
        self.current_state = None
        
    def _start_server(self):
        asyncio.set_event_loop(self.loop)
        start_server = websockets.serve(self._handler, "127.0.0.1", self.port)
        self.loop.run_until_complete(start_server)
        self.loop.run_forever()
        
    async def _handler(self, websocket):
        print("Game client connected!")
        try:
            async for message in websocket:
                msg_data = json.loads(message)
                msg_type = msg_data.get("type")
                
                if msg_type == "init":
                    self.iteration_count += 1
                    self.last_score = 0
                    self.last_coins = 0
                    continue
                
                elif msg_type == "save":
                    continue
                    
                elif msg_type == "state":
                    game_state = msg_data.get("data", {})
                    player = game_state.get('player', {})
                    dead = player.get('dead', False)
                    score = player.get('score', 0)
                    coins = player.get('coins', 0)
                    
                    if score > self.session_best_score:
                        self.session_best_score = score
                        
                    # Calculate reward
                    reward = 0.1  # Base survival bonus
                    reward += (score - self.last_score) * 0.5
                    reward += (coins - self.last_coins) * 2.0
                    if dead:
                        reward -= 10.0
                        
                    self.last_score = score
                    self.last_coins = coins
                    
                    state = extract_state(game_state)
                    
                    # Put state, reward, done in queue
                    self.state_queue.put((state, reward, dead))
                    
                    if dead:
                        # Client will disconnect/reconnect, don't wait for action
                        continue
                        
                    # Wait for action from Gym
                    action_idx = self.action_queue.get()
                    
                    response = {
                        "action": ACTIONS[action_idx],
                        "iteration": self.iteration_count,
                        "train_count": self.train_count,
                        "best_score": self.session_best_score,
                        "reward": reward
                    }
                    await websocket.send(json.dumps(response))
                    
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected.")
            
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # We wait for the next state from the queue, which should be the start of a new game
        # We might need to clear the queues first
        while not self.state_queue.empty():
            self.state_queue.get()
        while not self.action_queue.empty():
            self.action_queue.get()
            
        print("Waiting for new episode to start...")
        state, _, _ = self.state_queue.get()
        return state, {}

    def step(self, action):
        # Send action to websocket
        self.action_queue.put(action)
        
        # Wait for next state
        state, reward, done = self.state_queue.get()
        
        truncated = False
        info = {}
        
        return state, reward, done, truncated, info

    def render(self):
        pass

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.server_thread.join()
