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

    # Claw parameters
    CLAW_SPEED = 2
    CLAW_MASS = 10
    CLAW_MIN_Y = 256
    CLAW_HALFLENGTH = 24
    CLAW_LEFT_VERTICES = np.array([(0, 0), (CLAW_HALFLENGTH, 0), (CLAW_HALFLENGTH, CLAW_HALFLENGTH), (0, CLAW_HALFLENGTH)])
    CLAW_RIGHT_VERTICES = np.array([(0, 0), (CLAW_HALFLENGTH, 0), (CLAW_HALFLENGTH, CLAW_HALFLENGTH), (0, CLAW_HALFLENGTH)])
    CLAW_MOI = pymunk.moment_for_poly(CLAW_MASS, CLAW_LEFT_VERTICES.tolist())

    BALL_RADIUS = 24
    BALL_MASS = 1
    GOAL_RADIUS = 8
    FLOOR_Y = 32
    WALL_DISTANCE = 96
    WALL_SIZE = (32, 128)
    WALL_COLOR = (64,64,64)
    TARGET_DISTANCE_REQUIREMENT = 5
    GRAVITY = -256
    PHYSICS_TIMESTEP = 1/50


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
        return {"agent": self._claw_hinge.position, "ball": self._ball.body.position,
                "target": self._target_location, "claw_angle": self._ball.body.position}

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # The space for physics
        self._space = pymunk.Space()
        self._space.gravity = (0,self.GRAVITY)

        # Choose the agent's location uniformly at random
        agent_location = (np.array([self.window_size[0], self.window_size[1]]) * self.np_random.random(2, dtype=float)) \
                                + np.array([0, self.CLAW_MIN_Y])

        self._target_location = np.array([((self.window_size[0] - 2*self.BALL_RADIUS)* self.np_random.random(dtype=float)) + self.BALL_RADIUS, 
                                          self.FLOOR_Y+self.BALL_RADIUS])

        wall_locations = np.array([self._target_location[0] - self.WALL_DISTANCE, 
                                         self._target_location[0] + self.WALL_DISTANCE])
        
        # Place the ball to either side of the target
        left_offset = wall_locations[0] - self.WALL_SIZE[0]/2
        right_offset = wall_locations[1] + self.WALL_SIZE[0]/2
        left_space = left_offset - 2*self.BALL_RADIUS
        right_space = self.window_size[0] - right_offset - 2*self.BALL_RADIUS
        random_position = self.np_random.random(dtype=float) * (left_space + right_space)
        ball_x = (random_position + self.BALL_RADIUS) if random_position <= left_space else (random_position + right_offset - left_space + self.BALL_RADIUS)

        # Add the objects to the physical space
        ball = pymunk.Body(1, pymunk.moment_for_circle(self.BALL_MASS, 0, self.BALL_RADIUS), body_type=pymunk.Body.DYNAMIC)
        ball.position = (ball_x, self.FLOOR_Y+self.BALL_RADIUS)
        self._ball = pymunk.Circle(ball, self.BALL_RADIUS)
        self._space.add(ball, self._ball)

        wall_left = pymunk.Body(body_type=pymunk.Body.STATIC)
        wall_left.position = (wall_locations[0], self.FLOOR_Y + self.WALL_SIZE[1]/2)
        self._wall_left = pymunk.Poly.create_box(wall_left, self.WALL_SIZE)
        self._space.add(wall_left, self._wall_left)

        wall_right = pymunk.Body(body_type=pymunk.Body.STATIC)
        wall_right.position = (wall_locations[1], self.FLOOR_Y + self.WALL_SIZE[1]/2)
        self._wall_right = pymunk.Poly.create_box(wall_right, self.WALL_SIZE)
        self._space.add(wall_right, self._wall_right)

        floor = pymunk.Body(body_type=pymunk.Body.STATIC)
        floor.position = (self.window_size[0]/2, self.FLOOR_Y/2)
        self._floor = pymunk.Poly.create_box(floor, (self.window_size[0], self.FLOOR_Y))
        self._space.add(floor, self._floor)

        ##### Construct the claw #####
        # A point for the arms to hinge on.
        self._claw_hinge = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self._claw_hinge.position = agent_location.tolist()
        # The left claw arm
        claw_left = pymunk.Body(self.CLAW_MASS, self.CLAW_MOI, body_type=pymunk.Body.DYNAMIC)
        claw_left.position = self._claw_hinge.position
        self._claw_left = pymunk.Poly(claw_left, self.CLAW_LEFT_VERTICES.tolist())
        self._space.add(claw_left, self._claw_left)
        # Pivot joints make the arms rotate around the hinge
        pivot_joint_left = pymunk.PivotJoint(claw_left, self._claw_hinge, (0,0), (0,0))
        pivot_joint_left.error_bias = 0
        self._space.add(pivot_joint_left)
        # Rotary joints limit joints to control the opening and closing of the claw arms
        limit_joint_left = pymunk.RotaryLimitJoint(self._claw_hinge, claw_left, -np.pi/4, np.pi/4)
        self._space.add(limit_joint_left)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):

        # Move the claw
        velocity = self.CLAW_SPEED * self._action_to_direction[action[0]]
        self._claw_hinge.position = np.clip(self._claw_hinge.position + velocity, 0, self.window_size-1).tolist()

        self._space.step(self.PHYSICS_TIMESTEP)

        # An episode is done iff the agent has reached the target
        terminated = np.linalg.norm(self._ball.body.position - self._target_location) < self.TARGET_DISTANCE_REQUIREMENT

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
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size[0], self.window_size[1]))
            if self.clock is None:
                self.clock = pygame.time.Clock()

        if self.render_mode:
            canvas = pygame.Surface((self.window_size[0], self.window_size[1]))
            canvas.fill((255, 255, 255))

        # Floor
        pygame.draw.rect(canvas,
                         self.WALL_COLOR,
                         (0, 0, self.window_size[0], self.FLOOR_Y))

        # Walls
        pygame.draw.rect(canvas,
                         self.WALL_COLOR,
                         pygame.Rect(self._wall_left.body.position.x-self.WALL_SIZE[0]/2, 0, self.WALL_SIZE[0], self.WALL_SIZE[1]))
        pygame.draw.rect(canvas,
                         self.WALL_COLOR,
                         pygame.Rect(self._wall_right.body.position.x-self.WALL_SIZE[0]/2, 0, self.WALL_SIZE[0], self.WALL_SIZE[1]))

        # Ball
        pygame.draw.circle(canvas,
                           (255, 0, 0),
                           self._ball.body.position,
                           self.BALL_RADIUS)
        
        # Goal
        pygame.draw.circle(canvas,
                           (239, 191, 4),
                           self._target_location,
                           self.GOAL_RADIUS)

        # Claw
        pygame.draw.circle(canvas,
                           (239, 191, 4),
                           self._claw_hinge.position,
                           self.GOAL_RADIUS)
        pygame.draw.polygon(
            canvas,
            (0, 0, 255),
            np.array([self._claw_left.body.position]) + rotate_vertices(self.CLAW_LEFT_VERTICES,
                                                                    self._claw_left.body.angle)
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


# Rotate a list of 2D points around the origin by a given degrees
def rotate_vertices(vertices: np.ndarray, angle: float):
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return np.dot(vertices, rotation_matrix.T)