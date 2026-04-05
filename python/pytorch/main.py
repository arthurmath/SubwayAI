import json
import asyncio
import websockets
import logging
from ppo import Agent
import glob, os
import torch
from utils import RolloutBuffer, extract_state, save_plots, load_best, save_weights, weights_dir, device


# Configure logging to suppress noisy websocket handshake errors
logging.getLogger('websockets').setLevel(logging.ERROR)

# Left, Right, Jump, Slide, Nothing
ACTIONS = ['L', 'R', 'J', 'S', None]
UPDATE_TIMESTEP = 1000


agent = Agent(
    state_dim=16, 
    action_dim=len(ACTIONS), 
    lr_actor=0.0003, 
    lr_critic=0.001, 
    gamma=0.99, 
    epochs=10, 
    eps=0.2
)

session_best_score = 0
iteration_count = 0
train_count = 0
current_game_id = -1
global_buffer = RolloutBuffer()
global_timestep_count = 0
mean_reward = 0
last_mean_score = 0
ready_for_new_session = True  # True when server just started or after a training stop


# History tracking
scores_history = [] 
rewards_history = [] 
episode_scores = [] 
history_lock = asyncio.Lock()


async def perform_training(buffer, score):
    print(f"\nTraining PPO (buffer size: {len(buffer.states)})")
    global train_count, iteration_count, session_best_score, global_timestep_count, mean_reward, last_mean_score
    
    if len(buffer.states) == 0:
        return

    # Capture stats before buffer is cleared during training
    mean_reward = sum(buffer.rewards) / len(buffer.rewards) if buffer.rewards else 0
    best_reward_in_buffer = max(buffer.rewards) if buffer.rewards else 0
    
    await agent.train(buffer)
    
    async with history_lock:
        train_count += 1
        avg_score = sum(episode_scores) / len(episode_scores) if episode_scores else score
        last_mean_score = avg_score
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
    global session_best_score, iteration_count, train_count, current_game_id, global_timestep_count, ready_for_new_session
    
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
                mode = msg_data.get("mode", "train")
                game_id = msg_data.get("game_id", -1)
                warm_start = msg_data.get("warm_start", False)
                if mode == "train" and game_id != -1:
                    should_train = False
                    do_warm_start = False
                    async with history_lock:
                        if game_id != current_game_id:
                            current_game_id = game_id
                            iteration_count += 1
                            print(f"Game iteration: {iteration_count}")
                            if global_timestep_count >= UPDATE_TIMESTEP:
                                should_train = True
                            if warm_start and ready_for_new_session:
                                do_warm_start = True
                                ready_for_new_session = False
                    if should_train:
                        await perform_training(global_buffer, session_best_score)
                    if do_warm_start:
                        weights_file = msg_data.get("weights_file")
                        if weights_file:
                            path = os.path.join(weights_dir, weights_file)
                            state_dict = torch.load(path, map_location=device, weights_only=True)
                            agent.policy.load_state_dict(state_dict)
                            agent.policy_old.load_state_dict(state_dict)
                            print(f"Warm start with: {weights_file}")
                        else:
                            load_best(agent.policy, agent.policy_old)
                if mode == "ai":
                    weights_file = msg_data.get("weights_file")
                    if weights_file:
                        path = os.path.join(weights_dir, weights_file)
                        state_dict = torch.load(path, map_location=device, weights_only=True)
                        agent.policy.load_state_dict(state_dict)
                        agent.policy_old.load_state_dict(state_dict)
                        print(f"Loaded weights: {weights_file}")
                    else:
                        load_best(agent.policy, agent.policy_old)
                continue
                
            elif msg_type == "query_weights_list":
                files = glob.glob(f"{weights_dir}/score_*.pth")
                entries = []
                for f in files:
                    try:
                        base = os.path.basename(f)
                        parts = base.split("_")
                        score = int(parts[1])
                        raw_date = parts[2]  # YYYYMMDD
                        date_fmt = f"{raw_date[6:8]}/{raw_date[4:6]}/{raw_date[0:4]}"
                        entries.append({"filename": base, "score": score, "date": date_fmt})
                    except Exception:
                        pass
                entries.sort(key=lambda e: e["score"], reverse=True)
                await websocket.send(json.dumps({"weights": entries}))
                continue

            elif msg_type == "query_best_weights":
                files = glob.glob(f"{weights_dir}/score_*.pth")
                best_file = None
                best_score = -1
                for f in files:
                    try:
                        base = os.path.basename(f)
                        score = float(base.split("_")[1])
                        if score > best_score:
                            best_score = score
                            best_file = base
                    except Exception:
                        pass
                await websocket.send(json.dumps({ "filename": best_file or "" }))
                continue

            elif msg_type == "save":
                print(f"\nTraining stopped. Saving weights and plots.")
                save_weights(agent.policy, session_best_score)
                save_plots(scores_history, rewards_history)
                ready_for_new_session = True
                continue
                
            elif msg_type == "state":
                game_state = msg_data.get("data", {})
                player = game_state.get('player', {})
                obstacles = game_state.get('obstacles', [])
                dead = player.get('dead', False)
                score = player.get('score', 0)
                coins = player.get('coins', 0)
                
                if score > session_best_score:
                    session_best_score = score
                
                if mode == "train":
                    # If we took an action in the previous step, calculate its reward
                    if len(local_buffer.states) > len(local_buffer.rewards):

                        reward = (score - last_score) * 5.0 + (coins - last_coins) * 0.5

                        # # Danger penalty: penalize being in a lane with a close obstacle
                        # # that the current state/action didn't avoid.
                        # # Rolling avoids "high" obstacles, so exempt those.
                        # rolling = player.get('rolling', False)
                        # player_lane = player.get('lane', 1)
                        # close_obs = [o for o in obstacles
                        #     if o.get('lane') == player_lane and -20.0 < o.get('z', -999) < 0
                        # ]
                        # for o in close_obs:
                        #     if not (o.get('type') == 'high' and rolling):
                        #         reward -= 3.0
                        #         break

                        if dead:
                            reward -= 50.0
                        
                        local_buffer.rewards.append(reward)
                        local_buffer.is_terminals.append(dead)
                        current_reward = reward
                
                last_score = score
                last_coins = coins
                
                if dead and mode == "train":
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
                    action = agent.act_train(state, local_buffer)
                else:
                    action, probs = agent.act_play(state)
                
                response = {
                    "action": ACTIONS[action],
                    "iteration": int(iteration_count),
                    "train_count": int(train_count),
                    "best_score": float(session_best_score),
                    "reward": float(current_reward),
                    "avg_score": float(last_mean_score)
                }
                if mode == "ai":
                    response["probs"] = probs
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
    print(f"Starting Python AI Server on ws://127.0.0.1:{port}")
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