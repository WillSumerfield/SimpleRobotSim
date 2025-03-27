import gymnasium as gym
import numpy as np

from Grasper.envs.manipulation import Object


SQ2 = np.sqrt(2)


class BetterExploration(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.env = env

        # Subtask rewards, so we can track performance on each subtask
        self._subtask_rewards = {i: [] for i in range(self.env.unwrapped.OBJECT_TYPES)}
        self._subtask_counts = {i: 0 for i in range(self.env.unwrapped.OBJECT_TYPES)}
        self._subtask_sums = {i: 0 for i in range(self.env.unwrapped.OBJECT_TYPES)}

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._total_reward = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

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
        if terminated:
            reward = 1000
        self._total_reward += reward

        # Track the subtask performance at the end of the episode
        if terminated or truncated:
            object_type = self.env.unwrapped._object._type.value
            if self._subtask_counts[object_type] >= (100//self.env.unwrapped.OBJECT_TYPES):
                last_reward = self._subtask_rewards[object_type].pop(0)
                self._subtask_sums[object_type] -= last_reward
            else: 
                self._subtask_counts[object_type] += 1
            self._subtask_rewards[object_type].append(self._total_reward)
            self._subtask_sums[object_type] += self._total_reward

        return obs, reward, terminated, truncated, info
    
    def get_subtask_performance(self):
        subtask_rewards = {}
        for type_name, v in Object.ObjectTypes.__members__.items():
            value = v.value
            if self._subtask_counts[value] > 0:
                subtask_rewards[type_name] = self._subtask_sums[value] / self._subtask_counts[value]
            else:
                subtask_rewards[type_name] = 0
        return subtask_rewards