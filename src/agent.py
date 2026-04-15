import numpy as np
from stable_baselines3 import PPO
import os

class DroneAgent:
    def __init__(self, env, learning_rate=3e-4, discount_factor=0.99):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.model = PPO(
            "MlpPolicy", 
            env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=discount_factor,
            clip_range=0.2,
            verbose=0
        )

    def select_action(self, state):
        state = state.reshape(1, -1)
        action, _ = self.model.predict(state, deterministic=False)
        return int(action[0])

    def train(self, *args, **kwargs):
        # PPO training handled via learn() in batches
        pass

    def learn(self, total_timesteps, callback=None):
        return self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def save_model(self, filename):
        # SB3 saves to directory
        save_dir = filename.rsplit('.', 1)[0]  # remove .h5
        os.makedirs(save_dir, exist_ok=True)
        self.model.save(save_dir)

    def update_policy(self):
        pass

    def load_model(self, path):
        self.model = PPO.load(path)
