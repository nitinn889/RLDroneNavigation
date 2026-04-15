import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.agent import DroneAgent
from src.environment import DroneEnvironment

class TestDroneAgent(unittest.TestCase):

    def setUp(self):
        self.environment = DroneEnvironment()
        self.agent = DroneAgent(self.environment)

    def test_select_action(self):
        obs, _ = self.environment.reset()
        action = self.agent.select_action(obs)
        self.assertTrue(0 <= action < self.environment.action_space.n)

    def test_train(self):
        obs, _ = self.environment.reset()
        action = self.agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = self.environment.step(action)
        self.agent.train(obs, action, reward, next_obs, terminated)
        self.assertIsNotNone(self.agent.model)

    def test_update_policy(self):
        self.agent.update_policy()
        # PPO policy updated internally during learn

    def test_agent_behavior(self):
        obs, _ = self.environment.reset()
        total_reward = 0
        for _ in range(10):
            action = self.agent.select_action(obs)
            obs, reward, terminated, truncated, _ = self.environment.step(action)
            total_reward += reward
            self.agent.train(obs, action, reward, obs, terminated or truncated)
            if terminated or truncated:
                break
        self.assertTrue(total_reward > -1000)  # No massive penalty

if __name__ == '__main__':
    unittest.main()