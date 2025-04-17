import gymnasium as gym
import numpy as np
import pygame

from Grasper.envs.manipulation import WINDOW_SIZE, Object
from hand_morphologies import HAND_TYPES


SQ2 = np.sqrt(2)
PI2 = np.pi * 2


class BetterExploration(gym.Wrapper):

    CLOSED_PERCENTAGE = 0.6
    DIGIT_LENGTH_PERCENTAGE = 0.3

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.env.unwrapped._render_flip = False

        # Subtask rewards, so we can track performance on each subtask
        self._subtask_count = self.env.unwrapped.OBJECT_TYPES
        self._subtask_rewards = {i: [] for i in range(self._subtask_count)}
        self._subtask_counts = {i: 0 for i in range(self._subtask_count)}
        self._subtask_sums = {i: 0 for i in range(self._subtask_count)}

        self._near_target = False
        self._closed_hand = False

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._closed_angle = self.env.unwrapped._hand.parameters.rotation_max * self.CLOSED_PERCENTAGE / (np.pi*2)
        self._max_dist = (self.env.unwrapped._hand.max_digit_len*self.DIGIT_LENGTH_PERCENTAGE + self.env.unwrapped._hand.BASE_RADIUS + self.env.unwrapped._object.SIZE/2) / WINDOW_SIZE[0]
        self._xy_ratio = WINDOW_SIZE[1]/WINDOW_SIZE[0]
        self._total_reward = 0
        obs = self.modify_obs(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        obs = self.modify_obs(obs)

        # Use more iterative rewards to enable better exploration
        hand_pos = obs[self._subtask_count:self._subtask_count+4]
        obj_pos = obs[self._subtask_count+4:self._subtask_count+7]
        target_pos = obs[self._subtask_count+7:self._subtask_count+10]
        dist_to_obj = np.sqrt(obj_pos[0]**2 + (obj_pos[1]*self._xy_ratio)**2) / SQ2
        dist_to_target = np.sqrt(target_pos[0]**2 + (target_pos[1]*self._xy_ratio)**2) / SQ2
        self._near_target = np.abs(dist_to_obj) < self._max_dist
        self._closed_hand = hand_pos[2] > self._closed_angle and hand_pos[3] < -self._closed_angle
        
        # If the object has the target, reward the agent for moving the object to the target
        if self._near_target:
            if self._closed_hand:
                reward = (1-dist_to_target)**2
            else:
                reward = -0.1
        # If the hand is not near the object, reward the agent for moving the hand to the object
        else:
            if action[2] == 1:
                reward = -dist_to_obj
            else:
                reward = -1
        if terminated:
            reward = 1000 + (self.env.unwrapped.MAX_TIME - self.env.unwrapped._elapsed_steps)
        self._total_reward += reward

        # Track the subtask performance at the end of the episode
        if terminated or truncated:
            object_type = self.env.unwrapped._object._type
            if self._subtask_counts[object_type] >= (100//self.env.unwrapped.OBJECT_TYPES):
                last_reward = self._subtask_rewards[object_type].pop(0)
                self._subtask_sums[object_type] -= last_reward
            else: 
                self._subtask_counts[object_type] += 1
            self._subtask_rewards[object_type].append(self._total_reward)
            self._subtask_sums[object_type] += self._total_reward

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.render_mode:
            canvas = pygame.Surface((400, 20))
            canvas.fill((255, 255, 255))
            if self.render_mode == "rgb_array":
                prev_canvas = self.env.render()
                if self.env.unwrapped.window is None:
                    self.env.unwrapped.window = 1
                    pygame.init()

        # Draw the current reward
        font = pygame.font.Font(None, 24)
        text = font.render(f"Reward: {int(self._total_reward)}", True, (0, 0, 0))
        canvas.blit(text, (4, 4))

        text = font.render(f"Near: {'yes' if self._near_target else 'no'}", True, (0, 0, 0))
        canvas.blit(text, (120, 4))

        text = font.render(f"Closed: {'yes' if self._closed_hand else 'no'}", True, (0, 0, 0))
        canvas.blit(text, (240, 4))

        if self.render_mode == "human":
            self.env.unwrapped.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.env.unwrapped.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            # Draw the current canvas on top of the previous canvas
            new_canvas = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
            prev_canvas[0:new_canvas.shape[0], 0:new_canvas.shape[1], 0:new_canvas.shape[2]] = new_canvas
            return prev_canvas

    def get_subtask_performance(self):
        subtask_rewards = {}
        for type_name, v in Object.ObjectTypes.__members__.items():
            value = v.value
            if self._subtask_counts[value] > 0:
                subtask_rewards[type_name] = self._subtask_sums[value] / self._subtask_counts[value]
            else:
                subtask_rewards[type_name] = 0
        return subtask_rewards
    
    def modify_obs(self, obs):
        # Modify the obs to be relative to the agent
        hand_pos = obs[self._subtask_count:self._subtask_count+4]
        obj_pos = obs[self._subtask_count+4:self._subtask_count+7]
        target_pos = obs[self._subtask_count+7:self._subtask_count+10]
        obs[self._subtask_count+4:self._subtask_count+6] = obj_pos[:2] - hand_pos[:2]
        obs[self._subtask_count+7:self._subtask_count+9] = target_pos[:2] - hand_pos[:2]
        return obs
    

class HandParams(gym.Wrapper):
    def __init__(self, env, hand_type):
        super().__init__(env)
        self.env = env
        self.hand_params = HAND_TYPES[hand_type]

    def reset(self, seed=None, options=None):
        if not options:
            options = {}
        options["hand_parameters"] = self.hand_params
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info
    

class TaskType(gym.Wrapper):
    def __init__(self, env, task_type):
        super().__init__(env)
        self.env = env
        self.task_type = task_type

    def reset(self, seed=None, options=None):
        if not options:
            options = {}
        options["object_type"] = self.task_type
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info