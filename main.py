import asyncio
import websockets
import json
import numpy as np
from ppo import Agent

# Possible actions: 'L' (Left), 'R' (Right), 'J' (Jump), 'S' (Slide), or None (do nothing)
ACTIONS = ['L', 'R', 'J', 'S', None]
ACTION_DIM = len(ACTIONS)
STATE_DIM = 16
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
ppo_agent = Agent(
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
            otype = 1.0 if t == 'low' else (0.5 if t == 'high' else 0.0)
            state.extend([z_norm, otype])
            obs_by_lane[lane] = obs_z
        else:
            state.extend([1.0, 0.0]) # Z = 1.0 (very far), Type = 0.0
            obs_by_lane[lane] = -500.0 # No obstacle, very far
            
    # 3. Coins: Track next per lane
    for lane in range(3):
        coins_in_lane = [c for c in coins if c.get('lane') == lane and c.get('z', 0) < 0]
        coins_in_lane.sort(key=lambda c: c.get('z', 0), reverse=True)
        
        obs_z = obs_by_lane[lane]
        
        if coins_in_lane:
            first_coin = coins_in_lane[0]
            z_norm = abs(first_coin.get('z', 0)) / 50.0
            
            # Count coins before the next obstacle in this lane
            # Coins are before obstacle if their z is closer to 0 than obs_z (i.e., coin_z > obs_z)
            count = sum(1 for c in coins_in_lane if c.get('z', 0) > obs_z)
            state.extend([z_norm, float(count)])
        else:
            state.extend([1.0, 0.0]) # Dist 1.0, Count 0.0
            
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
                mode = msg_data.get("mode", "train") # récupère la valeur de la clé "mode" (défaut à "train" si clé absente)
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
                    action_idx = ppo_agent.predict_action(state_vec)
                
                response = {"action": ACTIONS[action_idx]}
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
            action_idx = np.random.choice(len(ACTIONS), p=probs)

            response = {"action": ACTIONS[action_idx]}
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