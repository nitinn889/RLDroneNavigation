# File: /drone-navigation-rl/drone-navigation-rl/src/main.py

import os
from stable_baselines3.common.callbacks import EvalCallback
from agent import DroneAgent
from environment import DroneEnvironment

def main():
    # Initialize the environment and agent
    env = DroneEnvironment()
    agent = DroneAgent(env)

    # PPO Training
    log_dir = "./ppo_drone_logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    eval_callback = EvalCallback(
        env, 
        best_model_save_path=f"{log_dir}/best/",
        log_path=log_dir, 
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True, 
        render=False
    )
    
    total_timesteps = 200000
    print("Starting PPO training...")
    agent.learn(total_timesteps=total_timesteps, callback=eval_callback)
    print("Training completed!")

    # Save the trained model
    agent.save_model("drone_agent_model")
    
    # Test the trained agent
    test_env = DroneEnvironment()
    obs, _info = test_env.reset()
    episode_reward = 0
    for step in range(1000):
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, _info = test_env.step(action)
        episode_reward += reward
        test_env.render()
        if terminated or truncated:
            print(f"Test episode reward: {episode_reward}")
            obs, _info = test_env.reset()
            episode_reward = 0
    test_env.close()

if __name__ == "__main__":
    main()