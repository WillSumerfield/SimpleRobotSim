from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import pymunk


# Constants
WINDOW_SIZE = np.array([512, 384])  # The size of the space
FLOOR_Y = 32
FLOOR_COLOR = (64, 64, 64)
PI2 = 2*np.pi
HPI = np.pi/2
DOME_PRECISION = 12

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

def get_dome_vertices(size):
    vertices = [(size, 0)]
    for i in range(DOME_PRECISION):
        angle1 = np.pi * (i+1) / DOME_PRECISION
        new_point = (size * np.cos(angle1), size * np.sin(angle1))
        vertices.append(new_point)
    return vertices


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
    MIN_Y_SPAWN = 256
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
        left_offset = rotate_vertices(np.array([0, self.SEGMENT_SIZE[1]]), self._segment_ul_body.angle)
        right_offset = rotate_vertices(np.array([0, self.SEGMENT_SIZE[1]]), self._segment_ur_body.angle)
        self._segment_ll_body = pymunk.Body(self.SEGMENT_MASS, self.SEGMENT_MOI, body_type=pymunk.Body.DYNAMIC)
        self._segment_ll_body.position = self._segment_ul_body.position + left_offset
        self._segment_ll_body.angle = self._segment_ul_body.angle + self.DIGIT_ANGLE
        self._segment_ll = pymunk.Poly(self._segment_ll_body, self.SEGMENT_VERTICES)
        self._segment_ll.friction = self.SEGMENT_FRICTION
        self._segment_ll.filter = self.SEGMENT_COLLISION_FILTER
        space.add(self._segment_ll_body, self._segment_ll)
        self._segment_lr_body = pymunk.Body(self.SEGMENT_MASS, self.SEGMENT_MOI, body_type=pymunk.Body.DYNAMIC)
        self._segment_lr_body.position = self._segment_ur_body.position + right_offset
        self._segment_lr_body.angle = self._segment_ur_body.angle - self.DIGIT_ANGLE
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

    class ObjectTypes(Enum):
        circle = 0
        square = 1
        dome   = 2
        sheet  = 3
        cross  = 4

    COLOR = (255, 0, 0)
    SIZE = 48
    MASS = 1
    FRICTION = 0.5
    SPAWN_X_BUFFER = 32

    CIRCLE_MOI = pymunk.moment_for_circle(MASS, 0, SIZE/2)
    SQUARE_MOI = pymunk.moment_for_box(MASS, (SIZE, SIZE))
    SHEET_HEIGHT = SIZE/4
    SHEET_MOI = pymunk.moment_for_box(MASS, (SIZE*2, SHEET_HEIGHT))
    DOME_VERTICES = get_dome_vertices(SIZE/2)
    DOME_MOI = pymunk.moment_for_poly(MASS, DOME_VERTICES)
    CROSS_WIDTH = SIZE/8
    CROSS_MOI = 2*pymunk.moment_for_box(MASS/2, (SIZE, CROSS_WIDTH))


    def __init__(self, space, rng):
        # Setup the object
        self._type = rng.choice(list(self.ObjectTypes))
        self._type_vec = np.zeros(len(self.ObjectTypes), dtype=float)
        self._type_vec[self._type.value] = 1
        if self._type == self.ObjectTypes.circle:
            self._body = pymunk.Body(1, self.CIRCLE_MOI, body_type=pymunk.Body.DYNAMIC)
            self._shape = pymunk.Circle(self._body, self.SIZE/2)
        elif self._type == self.ObjectTypes.square:
            self._body = pymunk.Body(1, self.SQUARE_MOI, body_type=pymunk.Body.DYNAMIC)
            self._shape = pymunk.Poly.create_box(self._body, (self.SIZE, self.SIZE))
        elif self._type == self.ObjectTypes.dome:
            self._body = pymunk.Body(1, self.CROSS_MOI, body_type=pymunk.Body.DYNAMIC)
            self._shape = pymunk.Poly(self._body, self.DOME_VERTICES)
        elif self._type == self.ObjectTypes.sheet:
            self._body = pymunk.Body(1, self.SHEET_MOI, body_type=pymunk.Body.DYNAMIC)
            self._shape = pymunk.Poly.create_box(self._body, (self.SIZE*2, self.SHEET_HEIGHT))
        else: # Cross
            self._body = pymunk.Body(1, self.CROSS_MOI, body_type=pymunk.Body.DYNAMIC)
            self._shape = pymunk.Poly.create_box(self._body, (self.SIZE, self.CROSS_WIDTH))
            self._shape2 = pymunk.Poly.create_box(self._body, (self.CROSS_WIDTH, self.SIZE))
            self._shape2.angle = np.pi/2
            self._shape2.friction = self.FRICTION
            self._shape2.filter = pymunk.ShapeFilter(categories=0b10, mask=pymunk.ShapeFilter.ALL_MASKS())

        # Set the object's properties
        self._body.position = ((WINDOW_SIZE[0]-2*self.SPAWN_X_BUFFER)*rng.random(dtype=float) + self.SPAWN_X_BUFFER, FLOOR_Y+self.SIZE)
        self._shape.friction = self.FRICTION
        self._shape.filter = pymunk.ShapeFilter(categories=0b10, mask=pymunk.ShapeFilter.ALL_MASKS())

        if self._type == self.ObjectTypes.cross:
            space.add(self._body, self._shape, self._shape2)
        else:
            space.add(self._body, self._shape)

    def get_state(self):
        return np.array([self._body.position[0]/WINDOW_SIZE[0], 
                        self._body.position[1]/WINDOW_SIZE[1], 
                        self._body.angle/PI2])
    
    def get_type(self):
        return self._type_vec
    
    def draw(self, canvas):
        if self._type == self.ObjectTypes.circle:
            pygame.draw.circle(canvas,
                               self.COLOR,
                               self._body.position,
                               self.SIZE/2)
        elif self._type == self.ObjectTypes.square:
            pygame.draw.polygon(canvas,
                                self.COLOR,
                                np.array([self._body.position]) +
                                rotate_vertices(np.array(get_rect_vertices((self.SIZE, self.SIZE)) - np.array([self.SIZE/2, self.SIZE/2])), self._body.angle))
        elif self._type == self.ObjectTypes.dome:
            pygame.draw.polygon(canvas,
                                self.COLOR,
                                np.array([self._body.position]) + rotate_vertices(self.DOME_VERTICES, self._body.angle))
        elif self._type == self.ObjectTypes.sheet:
            pygame.draw.polygon(canvas,
                                self.COLOR,
                                np.array([self._body.position]) + 
                                rotate_vertices(np.array(get_rect_vertices((self.SIZE*2, self.SHEET_HEIGHT)) - np.array([self.SIZE, self.SHEET_HEIGHT/2])), self._body.angle))
        else: # Cross
            pygame.draw.polygon(canvas,
                                self.COLOR,
                                np.array([self._body.position]) + rotate_vertices(self._shape.get_vertices(), self._body.angle))
            pygame.draw.polygon(canvas,
                                self.COLOR,
                                np.array([self._body.position]) + rotate_vertices(self._shape2.get_vertices(), self._body.angle))


class ManipulationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    GOAL_RADIUS = 32
    GOAL_ROTATION = 0.1
    GRAVITY = -256
    PHYSICS_TIMESTEP = 1/50
    TARGET_Y_BUFFER = 32
    MAX_TIME = 400

    AGENT_SPACE = 4 # x, y, digit_angle1, digit_angle2
    OBJECT_SPACE = 3 # x, y, angle
    TARGET_SPACE = 3 # x, y, angle
    OBJECT_TYPES = len(Object.ObjectTypes)
    OBS_SPACE = AGENT_SPACE + OBJECT_SPACE + TARGET_SPACE + OBJECT_TYPES


    def __init__(self, render_mode=None):
        super().__init__()
        
        # What the agent sees
        self.observation_space = spaces.Box(-1, 1, shape=(self.OBS_SPACE,), dtype=float)

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
        return np.concat((self._hand.get_state(), self._object.get_state(), self._obs_target_position, self._object.get_type()), dtype=float)
    
    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._elapsed_steps = 0

        self._space = pymunk.Space()
        self._space.gravity = (0, self.GRAVITY)
        self._space.collision_slop = 0.5

        # Add the objects to the physical space
        self._floor = Floor(self._space)
        self._hand = Hand(self._space, self.np_random)
        self._object = Object(self._space, self.np_random)

        self._target_position = np.array([((WINDOW_SIZE[0] - 2*self._object.SIZE)*self.np_random.random(dtype=float)) + self._object.SIZE, 
                                          ((WINDOW_SIZE[1] - 2*self._object.SIZE - self._hand.MIN_Y_SPAWN - self.TARGET_Y_BUFFER)*self.np_random.random(dtype=float)) + 
                                            self._object.SIZE + self._hand.MIN_Y_SPAWN,
                                            self.np_random.uniform(-np.pi, np.pi)])
        self._obs_target_position = self._target_position / np.array([WINDOW_SIZE[0], WINDOW_SIZE[1], np.pi])

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
        obj_outofbounds = self._object._body.position[0] < 0 or self._object._body.position[0] > WINDOW_SIZE[0] or self._object._body.position[1] > WINDOW_SIZE[1]
        timeout = self._elapsed_steps >= self.MAX_TIME
        truncated = obj_outofbounds or timeout

        # Check if goal is reached
        goal_dist = np.linalg.norm(np.array([self._object._body.position]) - self._target_position[:2])
        goal_angle_dist = 0 #np.abs(self._object._body.angle - self._target_position[2])
        terminated = bool((goal_dist <= self.GOAL_RADIUS) and (goal_angle_dist <= self.GOAL_ROTATION))

        reward = 100 if terminated else (-20 if obj_outofbounds else -1)
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
                self.window = pygame.display.set_mode((WINDOW_SIZE[0], WINDOW_SIZE[1]))
            if self.clock is None:
                self.clock = pygame.time.Clock()

        if self.render_mode:
            canvas = pygame.Surface((WINDOW_SIZE[0], WINDOW_SIZE[1]))
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