from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import pymunk


# Constants
WINDOW_SIZE = np.array([768, 512])  # The size of the space
FLOOR_Y = 32
FLOOR_COLOR = (64, 64, 64)
PI2 = 2*np.pi
HPI = np.pi/2

def get_rect_vertices(size):
    return [
        (0, 0),
        (0, size[1]),
        (size[0], size[1]),
        (size[0], 0)
    ]

def rotate_vertices(vertices: np.ndarray, angle: float):
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return np.dot(vertices, rotation_matrix.T)


class MoveActions(Enum):
    none = 0
    up = 1
    down = 2
    left = 3
    right = 4

class RotateActions(Enum):
    none = 0
    clockwise = 1
    counterclockwise = 2

class ClawActions(Enum):
    none = 0,
    open = 1,
    close = 2


class Hand():
    MOVE_SPEED = 3
    BASE_COLOR = (0, 0, 0)
    SEGMENT_COLOR = (128, 128, 128)
    BASE_RADIUS = 32
    MIN_Y_SPAWN = 64
    MAX_ANGLE = np.pi * (15/16.0)
    MIN_ANGLE = HPI
    DIGIT_ANGLE = np.pi/4
    SEGMENT_MASS = 10
    SEGMENT_SIZE = (8, 48)
    MIN_POS = np.array([0, FLOOR_Y+SEGMENT_SIZE[1]+BASE_RADIUS])
    MAX_POS = np.array([WINDOW_SIZE[0], WINDOW_SIZE[1]-1])
    SEGMENT_OFFSET = np.sqrt((BASE_RADIUS**2)/2)
    SEGMENT_VERTICES = get_rect_vertices(SEGMENT_SIZE)
    SEGMENT_MOI = pymunk.moment_for_poly(SEGMENT_MASS, [(0,0), (0,SEGMENT_SIZE[1]), (SEGMENT_SIZE[0],SEGMENT_SIZE[1]), (SEGMENT_SIZE[0],0)])
    SEGMENT_COLLISION_FILTER = pymunk.ShapeFilter(categories=0b100, mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b100)
    SEGMENT_FRICTION = 0.5
    FORCE = 10000


    def __init__(self, space, rng):
        position = ((np.array([WINDOW_SIZE[0], WINDOW_SIZE[1]-self.MIN_Y_SPAWN]) * rng.random(2, dtype=float)) \
                    + np.array([0, self.MIN_Y_SPAWN])).tolist()
        
        # Hinge
        self._hinge = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self._hinge.position = position
        self.hinge_shape = pymunk.Circle(self._hinge, self.BASE_RADIUS)
        self.hinge_shape.friction = self.SEGMENT_FRICTION
        self.hinge_shape.filter = self.SEGMENT_COLLISION_FILTER
        space.add(self._hinge, self.hinge_shape)

        # Digit upper segments
        self._segment_ul_body = pymunk.Body(self.SEGMENT_MASS, self.SEGMENT_MOI, body_type=pymunk.Body.DYNAMIC)
        self._segment_ul_body.position = self._hinge.position + (self.SEGMENT_OFFSET * np.array([-1, -1]))
        self._segment_ul_body.angle = self.MAX_ANGLE
        self._segment_ul = pymunk.Poly(self._segment_ul_body, self.SEGMENT_VERTICES)
        self._segment_ul.friction = self.SEGMENT_FRICTION
        self._segment_ul.filter = self.SEGMENT_COLLISION_FILTER
        space.add(self._segment_ul_body, self._segment_ul)
        self._segment_ur_body = pymunk.Body(self.SEGMENT_MASS, self.SEGMENT_MOI, body_type=pymunk.Body.DYNAMIC)
        self._segment_ur_body.position = self._hinge.position + (self.SEGMENT_OFFSET * np.array([1, -1]))
        self._segment_ur_body.angle = -self.MAX_ANGLE
        self._segment_ur = pymunk.Poly(self._segment_ur_body, self.SEGMENT_VERTICES)
        self._segment_ur.friction = self.SEGMENT_FRICTION
        self._segment_ur.filter = self.SEGMENT_COLLISION_FILTER
        space.add(self._segment_ur_body, self._segment_ur)

        # Digit lower segments
        self._segment_ll_body = pymunk.Body(self.SEGMENT_MASS, self.SEGMENT_MOI, body_type=pymunk.Body.DYNAMIC)
        self._segment_ll_body.position = self._segment_ul_body.position + (self.SEGMENT_SIZE[1] * np.array([0, -1]))
        self._segment_ll_body.angle = self.MAX_ANGLE
        self._segment_ll = pymunk.Poly(self._segment_ll_body, self.SEGMENT_VERTICES)
        self._segment_ll.friction = self.SEGMENT_FRICTION
        self._segment_ll.filter = self.SEGMENT_COLLISION_FILTER
        space.add(self._segment_ll_body, self._segment_ll)
        self._segment_lr_body = pymunk.Body(self.SEGMENT_MASS, self.SEGMENT_MOI, body_type=pymunk.Body.DYNAMIC)
        self._segment_lr_body.position = self._segment_ur_body.position + (self.SEGMENT_SIZE[1] * np.array([0, -1]))
        self._segment_lr_body.angle = -self.MAX_ANGLE
        self._segment_lr = pymunk.Poly(self._segment_lr_body, self.SEGMENT_VERTICES)
        self._segment_lr.friction = self.SEGMENT_FRICTION
        self._segment_lr.filter = self.SEGMENT_COLLISION_FILTER
        space.add(self._segment_lr_body, self._segment_lr)

        # Pivot joints make the digits rotate around the hinge
        self._pivot_ul = pymunk.PivotJoint(self._segment_ul_body, self._hinge, (self.SEGMENT_SIZE[0]/2,0), (-self.SEGMENT_OFFSET,-self.SEGMENT_OFFSET))
        self._pivot_ul.error_bias = 0
        self._pivot_ur = pymunk.PivotJoint(self._segment_ur_body, self._hinge, (self.SEGMENT_SIZE[0]/2,0), (self.SEGMENT_OFFSET,-self.SEGMENT_OFFSET))
        self._pivot_ur.error_bias = 0
        self._pivot_ll = pymunk.PivotJoint(self._segment_ll_body, self._segment_ul_body, (self.SEGMENT_SIZE[0]/2,0), (self.SEGMENT_SIZE[0]/2,self.SEGMENT_SIZE[1]))
        self._pivot_ll.error_bias = 0
        self._pivot_lr = pymunk.PivotJoint(self._segment_lr_body, self._segment_ur_body, (self.SEGMENT_SIZE[0]/2,0), (self.SEGMENT_SIZE[0]/2,self.SEGMENT_SIZE[1]))
        self._pivot_lr.error_bias = 0
        space.add(self._pivot_ul, self._pivot_ur, self._pivot_ll, self._pivot_lr)

        # Rotary joints limit joints to control the opening and closing of the digits
        self._limit_ul = pymunk.RotaryLimitJoint(self._hinge, self._segment_ul_body, self.MIN_ANGLE, self.MAX_ANGLE)
        self._limit_ur = pymunk.RotaryLimitJoint(self._hinge, self._segment_ur_body, -self.MAX_ANGLE, -self.MIN_ANGLE)
        self._limit_ll = pymunk.RotaryLimitJoint(self._segment_ll_body, self._segment_ul_body, -self.DIGIT_ANGLE, -self.DIGIT_ANGLE)
        self._limit_lr = pymunk.RotaryLimitJoint(self._segment_lr_body, self._segment_ur_body, self.DIGIT_ANGLE, self.DIGIT_ANGLE)
        space.add(self._limit_ul, self._limit_ur, self._limit_ll, self._limit_lr)

    def move(self, direction, rotation, open_hand):
        velocity = self.MOVE_SPEED * direction
        self._hinge.position = np.clip(np.array([self._hinge.position])+velocity, self.MIN_POS, self.MAX_POS)[-1].tolist()

        # Apply force to the digits
        force = self.FORCE if open_hand else -self.FORCE
        force_direction_l = np.array([np.cos(self._segment_ul_body.angle), np.sin(self._segment_ul_body.angle)]) * force
        force_direction_r = force_direction_l * np.array([-1, 1])
        force_position_l = rotate_vertices(np.array([self.SEGMENT_SIZE[0]/2, self.SEGMENT_SIZE[1]]), self._segment_ul_body.angle)
        force_position_r = force_position_l * np.array([-1, 1])
        self._segment_ul_body.apply_force_at_local_point(force_direction_l.tolist(), force_position_l.tolist())
        self._segment_ur_body.apply_force_at_local_point(force_direction_r.tolist(), force_position_r.tolist())

    def get_state(self):
        return np.array([self._hinge.position[0]/WINDOW_SIZE[0], 
                        self._hinge.position[1]/WINDOW_SIZE[1], 
                        self._segment_ul_body.angle/PI2,
                        self._segment_ur_body.angle/PI2])
 
    def draw(self, canvas):
        pygame.draw.circle(canvas,
                           self.BASE_COLOR,
                           self._hinge.position,
                           self.BASE_RADIUS)
        pygame.draw.polygon(canvas,
                            self.SEGMENT_COLOR,
                            np.array([self._segment_ul_body.position]) + rotate_vertices(self.SEGMENT_VERTICES, self._segment_ul_body.angle))
        pygame.draw.polygon(canvas,
                            self.SEGMENT_COLOR,
                            np.array([self._segment_ur_body.position]) + rotate_vertices(self.SEGMENT_VERTICES, self._segment_ur_body.angle))
        pygame.draw.polygon(canvas,
                            self.SEGMENT_COLOR,
                            np.array([self._segment_ll_body.position]) + rotate_vertices(self.SEGMENT_VERTICES, self._segment_ll_body.angle))
        pygame.draw.polygon(canvas,
                            self.SEGMENT_COLOR,
                            np.array([self._segment_lr_body.position]) + rotate_vertices(self.SEGMENT_VERTICES, self._segment_lr_body.angle))

class Floor():
    def __init__(self, space):
        self._body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self._body.position = (WINDOW_SIZE[0]/2, FLOOR_Y/2)
        self._floor = pymunk.Poly.create_box(self._body, (WINDOW_SIZE[0], FLOOR_Y))
        self._floor.friction = 1
        self._floor.filter = pymunk.ShapeFilter(categories=0b1, mask=pymunk.ShapeFilter.ALL_MASKS())
        space.add(self._body, self._floor)

    def draw(self, canvas):
        pygame.draw.rect(canvas, FLOOR_COLOR, (0, 0, WINDOW_SIZE[0], FLOOR_Y))

class Object():
    SIZE = 24
    MASS = 1
    FRICTION = 0.5
    SPAWN_X_BUFFER = 32


    def __init__(self, space, rng):
        self._body = pymunk.Body(1, pymunk.moment_for_circle(self.MASS, 0, self.SIZE), body_type=pymunk.Body.DYNAMIC)
        self._body.position = ((WINDOW_SIZE[0]-2*self.SPAWN_X_BUFFER)*rng.random(dtype=float) + self.SPAWN_X_BUFFER, FLOOR_Y+self.SIZE)
        self._shape = pymunk.Circle(self._body, self.SIZE)
        self._shape.friction = self.FRICTION
        self._shape.filter = pymunk.ShapeFilter(categories=0b10, mask=pymunk.ShapeFilter.ALL_MASKS())
        space.add(self._body, self._shape)

    def get_state(self):
        return np.array([self._body.position[0]/WINDOW_SIZE[0], 
                        self._body.position[1]/WINDOW_SIZE[1], 
                        self._body.angle/PI2])
    
    def draw(self, canvas):
        pygame.draw.circle(canvas,
                           (255, 0, 0),
                           self._body.position,
                           self.SIZE)


class ManipulationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    GOAL_RADIUS = 32
    GOAL_ROTATION = 0.1
    GRAVITY = -256
    PHYSICS_TIMESTEP = 1/50
    TARGET_Y_BUFFER = 32


    def __init__(self, render_mode=None):
        self.window_size = np.array([768, 512])  # The size of the PyGame window
        
        # What the agent sees
        self.observation_space = spaces.Dict(
            {
                "agent_state":  spaces.Box(0, 1, shape=(4,), dtype=float), # Agent x,y,digit1_angle,digit2_angle
                "object_state": spaces.Box(0, 1, shape=(3,), dtype=float), # Object x,y,theta
                "target_state": spaces.Box(0, 1, shape=(3,), dtype=float), # Target x,y,theta
            }
        )

        # What the agent can do
        self.action_space = spaces.MultiDiscrete([5, 3, 2]) # 5 movement, 3 rotation, 2 claw

        self._action_to_direction = {
            MoveActions.none.value:  np.array([0,  0]),
            MoveActions.right.value: np.array([1,  0]),
            MoveActions.up.value:    np.array([0,  1]),
            MoveActions.left.value:  np.array([-1, 0]),
            MoveActions.down.value:  np.array([0, -1])
        }

        self._action_to_rotation = {
            RotateActions.none.value:           0,
            RotateActions.clockwise.value:      1,
            RotateActions.counterclockwise.value: -1
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
        return  {
                    "agent_state": self._hand.get_state(), 
                    "object_state": self._object.get_state(),
                    "target_state": self._target_position
                }

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._elapsed_steps = 0

        self._space = pymunk.Space()
        self._space.gravity = (0, self.GRAVITY)

        # Add the objects to the physical space
        self._floor = Floor(self._space)
        self._hand = Hand(self._space, self.np_random)
        self._object = Object(self._space, self.np_random)

        self._target_position = np.array([((self.window_size[0] - 2*self._object.SIZE)*self.np_random.random(dtype=float)) + self._object.SIZE, 
                                          ((self.window_size[1] - 2*self._object.SIZE - self._hand.MIN_Y_SPAWN - self.TARGET_Y_BUFFER)*self.np_random.random(dtype=float)) + 
                                            self._object.SIZE + self._hand.MIN_Y_SPAWN,
                                            self.np_random.uniform(-np.pi, np.pi)])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        return observation, info

    def step(self, action):
        self._elapsed_steps += 1

        # Physics
        self._hand.move(self._action_to_direction[action[0]], self._action_to_rotation[action[1]], action[2] == ClawActions.open.value)
        self._space.step(self.PHYSICS_TIMESTEP)

        # Check if terminated
        obj_outofbounds = self._object._body.position[0] < 0 or self._object._body.position[0] > self.window_size[0] or self._object._body.position[1] > self.window_size[1]
        timeout = self._elapsed_steps >= 1000
        truncated = obj_outofbounds or timeout

        # Check if goal is reached
        goal_dist = np.linalg.norm(np.array([self._object._body.position]) - self._target_position[:2])
        goal_angle_dist = 0 #np.abs(self._object._body.angle - self._target_position[2])
        terminated = (goal_dist <= self.GOAL_RADIUS) and (goal_angle_dist <= self.GOAL_ROTATION)

        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        return observation, reward, terminated, truncated, info

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

        self._floor.draw(canvas)
        self._object.draw(canvas)
        self._hand.draw(canvas)
        pygame.draw.circle(canvas, (0, 255, 0), self._target_position[:2], self.GOAL_RADIUS)

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
