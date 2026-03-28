import json
import asyncio
import websockets
from ppo import Agent
from utils import RolloutBuffer, extract_state


# Left, Right, Jump, Slide, Nothing
ACTIONS = ['L', 'R', 'J', 'S', None]
ACTION_DIM = len(ACTIONS)
STATE_DIM = 16
UPDATE_TIMESTEP = 2000




agent = Agent(
    state_dim=STATE_DIM, 
    action_dim=ACTION_DIM, 
    lr_actor=0.0003, 
    lr_critic=0.001, 
    gamma=0.99, 
    epochs=4, 
    eps_clip=0.2
)

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
                    agent.load_best()
                continue
                
            elif msg_type == "save":
                print(f"Received save request. Saving weights with best score: {session_best_score}")
                agent.save(session_best_score)
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
                            await agent.train(local_buffer)
                            print("Training complete.")
                
                last_score = score
                last_coins = coins
                
                if dead and mode == "train":
                    # Agent died, JS client will auto-restart and establish a new connection.
                    continue
                    
                state = extract_state(game_state)
                
                if mode == "train":
                    action = agent.select_action(state, local_buffer)
                else:
                    # In 'ai player' mode, don't store in buffer
                    action = agent.select_action(state)
                
                response = {"action": ACTIONS[action]}
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
                await agent.train(local_buffer)




    

async def main():
    port = 8765
    print(f"Starting Python PPO AI Server on ws://localhost:{port}")
    async with websockets.serve(play_game, "localhost", port):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())