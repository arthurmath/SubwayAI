import asyncio
import websockets
import json
import torch

# Possible actions: 'L' (Left), 'R' (Right), 'J' (Jump), 'S' (Slide), or None (do nothing)
ACTIONS = ['L', 'R', 'J', 'S', None]

async def play_game(websocket):
    print("New connection from game client!")
    try:
        async for message in websocket:
            # 1. Receive game state from JS
            game_state = json.loads(message)
            
            # Use PyTorch to generate a random action
            # Creating a tensor of random integers to pick an action index
            action_idx = torch.randint(0, len(ACTIONS), (1,)).item()
            action = ACTIONS[action_idx]
            
            # 3. Send action back to JS
            response = {"action": action}
            await websocket.send(json.dumps(response))
            
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")

async def main():
    port = 8765
    print(f"Starting Python AI Server on ws://localhost:{port}")
    async with websockets.serve(play_game, "localhost", port):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())