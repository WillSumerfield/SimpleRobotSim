import gymnasium as gym
import numpy as np

SQ2 = np.sqrt(2)

class BetterExploration(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # Modify the obs to be relative to the agent
        obs[4:6] = obs[4:6] - obs[:2]
        obs[7:9] = obs[7:9] - obs[:2]

        # Use more iterative rewards to enable better exploration
        dist_to_obj = np.linalg.norm(obs[4:6]) / SQ2
        dist_to_target = np.linalg.norm(obs[7:9]) / SQ2
        near_target = np.abs(obs[4]) < 0.05 and np.abs(obs[5]) < 0.25
        closed_hand = obs[2] > 0.4 and obs[3] < -0.4
        
        # If the object has the target, reward the agent for moving the object to the target
        if near_target:
            if closed_hand:
                reward = (1-dist_to_target)
            else:
                reward = -0.1
        # If the hand is not near the object, reward the agent for moving the hand to the object
        else:
            if action[2] == 1:
                reward = -dist_to_obj
            else:
                reward = -1
        if done:
            reward = 1000
        
        return obs, reward, done, truncated, info