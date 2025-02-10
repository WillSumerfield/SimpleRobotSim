from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import pymunk


class MoveActions(Enum):
    up = 0
    down = 1
    left = 2
    right = 3

class ClawActions(Enum):
    open = 0
    close = 1

class GrasperEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    SPEED = 1
    BALL_RADIUS = 24
    GOAL_RADIUS = 8
    FLOOR_Y = 32
    CLAW_MIN_Y = 256
    WALL_DISTANCE = 96
    WALL_SIZE = (32, 128)
    
    TARGET_DISTANCE_REQUIREMENT = 5

    WALL_COLOR = (64,64,64)


    def __init__(self, render_mode=None):
        self.window_size = np.array([768, 512])  # The size of the PyGame window
        
        # What the agent sees
        self.observation_space = spaces.Dict(
            {
                "agent":  spaces.Box(0, self.window_size[0]-1, shape=(2,), dtype=float), # Agent Location
                "ball":   spaces.Box(0, self.window_size[0]-1, shape=(2,), dtype=float), # Ball Location
                "target": spaces.Box(0, self.window_size[0]-1, shape=(2,), dtype=float), # Target Location
                "claw_angle": spaces.Box(0, 180, shape=(1,), dtype=float),
            }
        )

        # What the agent can do
        self.action_space = spaces.MultiDiscrete([4, 2]) # 4 movement, 2 claw

        self._action_to_direction = {
            MoveActions.right.value: np.array([1,  0]),
            MoveActions.up.value:    np.array([0,  1]),
            MoveActions.left.value:  np.array([-1, 0]),
            MoveActions.down.value:  np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "ball": np.array([1.0,0]),
                "target": self._target_location, "claw_angle": np.array([1.0,0])}

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = (np.array([self.window_size[0], self.window_size[1]]) * self.np_random.random(2, dtype=float)) \
                                + np.array([0, self.CLAW_MIN_Y])

        self._target_location = np.array([((self.window_size[0] - 2*self.BALL_RADIUS)* self.np_random.random(dtype=float)) + self.BALL_RADIUS, 
                                          self.FLOOR_Y+self.BALL_RADIUS])

        self._wall_locations = np.array([self._target_location[0] - self.WALL_DISTANCE, 
                                         self._target_location[0] + self.WALL_DISTANCE])
        
        # Place the ball to either side of the target
        left_offset = self._wall_locations[0] - self.WALL_SIZE[0]/2
        right_offset = self._wall_locations[1] + self.WALL_SIZE[0]/2
        left_space = left_offset - 2*self.BALL_RADIUS
        right_space = self.window_size[0] - right_offset - 2*self.BALL_RADIUS
        random_position = self.np_random.random(dtype=float) * (left_space + right_space)
        ball_x = (random_position + self.BALL_RADIUS) if random_position <= left_space else (random_position + right_offset - left_space + self.BALL_RADIUS)
        self._ball_location = np.array([ball_x, self.FLOOR_Y+self.BALL_RADIUS])
        print(ball_x, random_position)
        print(left_offset, right_offset)
        print(left_space, right_space)
        print(self._wall_locations, "\n")

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
   
        velocity = self.SPEED * self._action_to_direction[action[0]]

        # Stop the agent from leaving the room
        self._agent_location = np.clip(self._agent_location + velocity, 0, self.window_size-1)

        # An episode is done iff the agent has reached the target
        terminated = np.linalg.norm(self._ball_location - self._target_location) < 5

        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size[0], self.window_size[1]))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size[0], self.window_size[1]))
        canvas.fill((255, 255, 255))

        # Floor
        pygame.draw.rect(canvas,
                         self.WALL_COLOR,
                         (0, 0, self.window_size[0], self.FLOOR_Y))

        # Walls
        pygame.draw.rect(canvas,
                         self.WALL_COLOR,
                         pygame.Rect(self._wall_locations[0]-self.WALL_SIZE[0]/2, 0, self.WALL_SIZE[0], self.WALL_SIZE[1]))
        pygame.draw.rect(canvas,
                         self.WALL_COLOR,
                         pygame.Rect(self._wall_locations[1]-self.WALL_SIZE[0]/2, 0, self.WALL_SIZE[0], self.WALL_SIZE[1]))

        # Ball
        pygame.draw.circle(canvas,
                           (255, 0, 0),
                           self._ball_location,
                           self.BALL_RADIUS)
        
        # Goal
        pygame.draw.circle(canvas,
                           (239, 191, 4),
                           self._target_location,
                           self.GOAL_RADIUS)

        # Claw
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            self._agent_location,
            self.BALL_RADIUS,
        )

        inv_canvas = pygame.transform.flip(canvas, False, True) # Invert Y
        if self.render_mode == "human":
            self.window.blit(inv_canvas, inv_canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(inv_canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
