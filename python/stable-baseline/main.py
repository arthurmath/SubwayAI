import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from env import SubwaySurfersEnv

class TrainCountCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(TrainCountCallback, self).__init__(verbose)
        self.custom_env = env

    def _on_step(self) -> bool:
        # Update the train count so the JS client can display it
        self.custom_env.train_count = self.num_timesteps
        return True

def main():
    # Create the environment
    env = SubwaySurfersEnv()
    
    # Create the PPO agent
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        tensorboard_log="./ppo_subwaysurfers_tensorboard/"
    )
    
    callback = TrainCountCallback(env)
    
    print("Starting training...")
    try:
        model.learn(total_timesteps=1000000, callback=callback)
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        # Save the model
        os.makedirs("models", exist_ok=True)
        model.save("models/ppo_subwaysurfers")
        print("Model saved to models/ppo_subwaysurfers.zip")
        env.close()

if __name__ == "__main__":
    main()
