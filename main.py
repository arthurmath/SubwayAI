import asyncio
import websockets
import json
import numpy as np
from ppo import PPOAgent

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Possible actions: 'L' (Left), 'R' (Right), 'J' (Jump), 'S' (Slide), or None (do nothing)
ACTIONS = ['L', 'R', 'J', 'S', None]
ACTION_DIM = len(ACTIONS)
STATE_DIM = 17
UPDATE_TIMESTEP = 2000




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



# Global PPO Agent instance
ppo_agent = PPOAgent(
    state_dim=STATE_DIM, 
    action_dim=ACTION_DIM, 
    lr_actor=0.0003, 
    lr_critic=0.001, 
    gamma=0.99, 
    K_epochs=4, 
    eps_clip=0.2
)

def extract_state(game_state):
    player = game_state.get('player', {})
    obstacles = game_state.get('obstacles', [])
    coins = game_state.get('coins', [])
    speed = game_state.get('speed', 10.0)
    
    # 1. Player state
    state = [
        player.get('lane', 1) / 2.0,
        player.get('y', 0) / 3.0,
        player.get('vy', 0) / 10.0,
        speed / 10.0
    ]
    
    # 2. Obstacles: filter z < 2.0 and sort by z descending (closest ahead first)
    obs_ahead = [o for o in obstacles if o.get('z', 0) < 2.0]
    obs_ahead.sort(key=lambda o: o.get('z', 0), reverse=True)
    
    for i in range(3):
        if i < len(obs_ahead):
            obs = obs_ahead[i]
            lane = obs.get('lane', 1) / 2.0
            z = obs.get('z', 0) / 50.0
            t = obs.get('type', 'low')
            otype = 0.5 if t == 'low' else (1.0 if t == 'high' else 0.0)
            state.extend([lane, z, otype])
        else:
            state.extend([0.0, 0.0, 0.0])
            
    # 3. Coins: filter z < 2.0, sort by z descending
    coins_ahead = [c for c in coins if c.get('z', 0) < 2.0]
    coins_ahead.sort(key=lambda c: c.get('z', 0), reverse=True)
    
    for i in range(2):
        if i < len(coins_ahead):
            c = coins_ahead[i]
            state.extend([c.get('lane', 1) / 2.0, c.get('z', 0) / 50.0])
        else:
            state.extend([0.0, 0.0])
            
    return state

session_best_score = 0

async def play_game(websocket):
    global session_best_score
    print("New connection from game client!")
    
    local_buffer = RolloutBuffer()
    last_score = 0
    last_coins = 0
    timestep = 0
    mode = "train"
    
    try:
        async for message in websocket:
            msg_data = json.loads(message)
            msg_type = msg_data.get("type")
            
            if msg_type == "init":
                mode = msg_data.get("mode", "train")
                print(f"Client initialized in mode: {mode}")
                if mode == "ai":
                    ppo_agent.load_best()
                continue
                
            elif msg_type == "save":
                print(f"Received save request. Saving weights with best score: {session_best_score}")
                ppo_agent.save(session_best_score)
                continue
                
            elif msg_type == "state":
                game_state = msg_data.get("data", {})
                player = game_state.get('player', {})
                dead = player.get('dead', False)
                score = player.get('score', 0)
                coins = player.get('coins', 0)
                
                if score > session_best_score:
                    session_best_score = score
                
                if mode == "train":
                    # If we took an action in the previous step, calculate its reward
                    if len(local_buffer.states) > len(local_buffer.rewards):
                        reward = 0.1  # Base survival bonus
                        reward += (score - last_score) * 0.5
                        reward += (coins - last_coins) * 2.0
                        if dead:
                            reward -= 10.0
                        
                        local_buffer.rewards.append(reward)
                        local_buffer.is_terminals.append(dead)
                        
                        timestep += 1
                        
                        # Perform PPO update if we have enough experiences
                        if timestep % UPDATE_TIMESTEP == 0:
                            print(f"Training PPO (local timestep {timestep})...")
                            await ppo_agent.update(local_buffer)
                            print("Training complete.")
                
                last_score = score
                last_coins = coins
                
                if dead and mode == "train":
                    # Agent died, JS client will auto-restart and establish a new connection.
                    continue
                    
                state_vec = extract_state(game_state)
                
                if mode == "train":
                    action_idx = ppo_agent.select_action(state_vec, local_buffer)
                else:
                    # In 'ai' mode, just act greedily or sample, but don't store in buffer
                    with torch.no_grad():
                        state_t = torch.FloatTensor(state_vec).to(device)
                        action, _, _ = ppo_agent.policy_old.act(state_t)
                        action_idx = action.item()
                
                action = ACTIONS[action_idx]
                
                response = {"action": action}
                await websocket.send(json.dumps(response))
            
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        print("Client disconnected.")
        if mode == "train":
            # Ensure states and rewards stay aligned if disconnected before assigning a reward
            if len(local_buffer.states) > len(local_buffer.rewards):
                local_buffer.states.pop()
                local_buffer.actions.pop()
                local_buffer.logprobs.pop()
                local_buffer.state_values.pop()
                
            # Optional: Train on remaining experiences before throwing away buffer
            if len(local_buffer.states) > 0:
                await ppo_agent.update(local_buffer)





async def play_random(websocket):
    print("New connection from game client!")
    try:
        async for message in websocket:
            # 1. Receive game state from JS
            game_state = json.loads(message)

            # Manually set custom probability array for actions.
            probs = np.array([0.1, 0.1, 0.1, 0.1, 0.6], dtype=np.float32)
            probs = probs / probs.sum()
            action_idx = np.random.choice(len(ACTIONS), p=probs)
            action = ACTIONS[action_idx]

            response = {"action": action}
            await websocket.send(json.dumps(response))

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")


    

async def main():
    port = 8765
    print(f"Starting Python PPO AI Server on ws://localhost:{port}")
    async with websockets.serve(play_game, "localhost", port):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())