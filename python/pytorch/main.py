import json
import asyncio
import websockets
import logging
from ppo import Agent
from utils import RolloutBuffer, extract_state, save_plots


# Configure logging to suppress noisy websocket handshake errors
logging.getLogger('websockets').setLevel(logging.ERROR)

# Left, Right, Jump, Slide, Nothing
ACTIONS = ['L', 'R', 'J', 'S', None]
UPDATE_TIMESTEP = 2000


agent = Agent(
    state_dim=16, 
    action_dim=len(ACTIONS), 
    lr_actor=0.0003, 
    lr_critic=0.001, 
    gamma=0.99, 
    epochs=4, 
    eps_clip=0.2
)

session_best_score = 0
iteration_count = 0
train_count = 0
current_game_id = -1
global_buffer = RolloutBuffer()
global_timestep_count = 0


# History tracking
scores_history = [] 
rewards_history = [] 
episode_scores = [] 
history_lock = asyncio.Lock()


async def perform_training(buffer, score):
    global train_count, iteration_count, session_best_score, global_timestep_count
    
    if len(buffer.states) == 0:
        return
    
    print(f"\nTraining PPO (buffer size: {len(buffer.states)})")
    await agent.train(buffer)

    # Capture stats before buffer is cleared during training
    mean_reward = sum(buffer.rewards) / len(buffer.rewards) if buffer.rewards else 0
    best_reward_in_buffer = max(buffer.rewards) if buffer.rewards else 0
    
    async with history_lock:
        train_count += 1
        avg_score = sum(episode_scores) / len(episode_scores) if episode_scores else score
        scores_history.append({
            'iteration': iteration_count,
            'avg_score': avg_score,
            'best_score': session_best_score
        })
        rewards_history.append({
            'iteration': iteration_count,
            'avg_reward': mean_reward,
            'best_reward': best_reward_in_buffer
        })
        episode_scores.clear()
        global_timestep_count = 0
    print(f"Training complete. Total trainings: {train_count}\n")


async def play_game(websocket):
    global session_best_score, iteration_count, train_count, current_game_id, global_timestep_count
    
    local_buffer = RolloutBuffer()
    last_score = 0
    last_coins = 0
    mode = "train"
    current_reward = 0
    
    try:
        async for message in websocket:
            msg_data = json.loads(message)
            msg_type = msg_data.get("type")
            
            if msg_type == "init":
                mode = msg_data.get("mode", "train") # récupère la valeur de la clé "mode" (défaut à "train" si clé absente)
                game_id = msg_data.get("game_id", -1)
                # print(f"Client initialized in mode: {mode}")
                if mode == "train" and game_id != -1:
                    should_train = False
                    async with history_lock:
                        if game_id != current_game_id:
                            current_game_id = game_id
                            iteration_count += 1
                            print(f"Game iteration: {iteration_count}")
                            if global_timestep_count >= UPDATE_TIMESTEP:
                                should_train = True
                    if should_train:
                        await perform_training(global_buffer, session_best_score)
                if mode == "ai":
                    agent.load_best()
                continue
                
            elif msg_type == "save":
                print(f"Training stopped. Saving weights and plots.")
                agent.save(session_best_score)
                save_plots(scores_history, rewards_history)
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
                        if not dead:
                            reward = 0.1  # Base survival bonus
                        else:
                            reward = -10.0
                        reward += (score - last_score) * 0.5
                        reward += (coins - last_coins) * 2.0
                            
                        
                        local_buffer.rewards.append(reward)
                        local_buffer.is_terminals.append(dead)
                        current_reward = reward
                
                last_score = score
                last_coins = coins
                
                if dead and mode == "train":
                    # Store the score for averaging
                    async with history_lock:
                        episode_scores.append(score)
                        # Move local experiences to global buffer
                        global_buffer.extend(local_buffer)
                        global_timestep_count += len(local_buffer.states)
                        local_buffer.clear()
                    # Agent died, JS client will auto-restart and establish a new connection.
                    continue
                    
                state = extract_state(game_state)
                
                if mode == "train":
                    action = agent.select_action(state, local_buffer)
                else:
                    # In 'ai player' mode, don't store in buffer
                    action = agent.select_action(state)
                
                response = {
                    "action": ACTIONS[action],
                    "iteration": int(iteration_count),
                    "train_count": int(train_count),
                    "best_score": float(session_best_score),
                    "reward": float(current_reward)
                }
                await websocket.send(json.dumps(response))
            
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        # print("Client disconnected.")
        if mode == "train":
            # Ensure states and rewards stay aligned if disconnected before assigning a reward
            if len(local_buffer.states) > len(local_buffer.rewards):
                local_buffer.states.pop()
                local_buffer.actions.pop()
                local_buffer.logprobs.pop()
                local_buffer.state_values.pop()
                
            # Move remaining local experiences to global buffer
            if len(local_buffer.states) > 0:
                async with history_lock:
                    global_buffer.extend(local_buffer)
                    global_timestep_count += len(local_buffer.states)
                    local_buffer.clear()






async def main():
    port = 8765
    print(f"Starting Python PPO AI Server on ws://127.0.0.1:{port}")
    try:
        async with websockets.serve(play_game, "127.0.0.1", port):
            await asyncio.Future()  # run forever
    except websockets.exceptions.ConnectionClosed:
        pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped")