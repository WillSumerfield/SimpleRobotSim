from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class MoveActions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3

class ClawActions(Enum):
    open = 0
    close = 1

class GrasperEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None):
        self.window_size = (768, 512)  # The size of the PyGame window
        
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
        self._agent_location = self.window_size[0] * self.np_random.random(2, dtype=float)

        # We will sample the target's location randomly until it does not
        # coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.window_size[0], size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action[0]]

        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.window_size[0] - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
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

        # First we draw the target
        pygame.draw.circle(canvas,
                           (255, 0, 0),
                           self._target_location,
                           10)

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            self._agent_location,
            10,
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
