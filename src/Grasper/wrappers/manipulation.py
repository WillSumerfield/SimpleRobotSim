import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TransformObservation
import numpy as np

SQ2 = np.sqrt(2)

class BetterExploration(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # Modify the obs to be relative to the agent
        #print(obs["agent_state"][:2], obs["object_state"][:2], obs["target_state"][:2], end='\r')
        obs[4:6] = obs[4:6] - obs[:2]
        obs[7:9] = obs[7:9] - obs[:2]
        obs[:2] = 0

        # Use more iterative rewards to enable better exploration
        dist_to_obj = np.linalg.norm(obs[4:6]) / SQ2
        dist_to_target = np.linalg.norm(obs[7:9]) / SQ2
        
        reward = -dist_to_obj if dist_to_obj > 0.15 else (1-dist_to_target)
        if done:
            reward = 1000
        
        return obs, reward, done, truncated, info