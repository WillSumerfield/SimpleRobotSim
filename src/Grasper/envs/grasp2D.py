from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import pymunk


# Constants
WINDOW_SIZE = np.array([256, 256])  # The size of the space
XY_RATIO = WINDOW_SIZE[1]/WINDOW_SIZE[0]
FLOOR_Y = 32
FLOOR_COLOR = (64, 64, 64)
PI2 = 2*np.pi
HPI = np.pi/2
SQ2 = np.sqrt(2)
DOME_PRECISION = 12
RANDOM_PRECISION = 12



def get_rect_vertices(size, offset=(0, 0)):
    return [
        (offset[0], offset[1]),
        (size[0] + offset[0], offset[1]),
        (size[0] + offset[0], size[1] + offset[1]),
        (offset[0], size[1] + offset[1])
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

class HandActions(Enum):
    none = 0
    open = 1
    close = 2


class Hand():

    class Parameters():
        def __init__(self, params: np.ndarray):
            self.segment_lengths = params[0:4] # TopL, TopR, BottomL, BottomR
            self.joint_angle = params[4:6] # Left, Right
            self.rotation_max = params[6]

    DEFAULT_PARAMETERS = Parameters(np.concatenate((
        np.array([48]*4), # Segment lengths
        np.array([np.pi/4]*2), # Joint angles
        np.array([np.pi * (15/16.0)]) # Rotation Max
    )))

    MOVE_SPEED = 3
    BASE_COLOR = (0, 0, 0)
    SEGMENT_COLOR = (128, 128, 128)
    BASE_RADIUS = 32
    MIN_Y_SPAWN = 192
    MIN_ANGLE = HPI
    SEGMENT_MASS = 10
    SEGMENT_WIDTH = 8
    SEGMENT_OFFSET = np.sqrt((BASE_RADIUS**2)/2)
    SEGMENT_L_COLLISION_FILTER = pymunk.ShapeFilter(categories=0b100, mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b100)
    SEGMENT_R_COLLISION_FILTER = pymunk.ShapeFilter(categories=0b1000, mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1000)
    SEGMENT_COLLISION_FILTER = pymunk.ShapeFilter(categories=0b1100, mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b1100)
    SEGMENT_FRICTION = 0.5
    FORCE = 20000


    def __init__(self, space, rng, parameters=None, centered=False):
        position = (WINDOW_SIZE[0]/2, self.MIN_Y_SPAWN) if centered else (rng.random()*WINDOW_SIZE[0], self.MIN_Y_SPAWN)#((np.array([WINDOW_SIZE[0], WINDOW_SIZE[1]-self.MIN_Y_SPAWN]) * rng.random(2, dtype=float)) \
                   #+ np.array([0, self.MIN_Y_SPAWN])).tolist()
        
        self.parameters = self.DEFAULT_PARAMETERS if parameters is None else self.Parameters(parameters)
        
        max_l_length = max(np.cos(self.parameters.joint_angle[0])*self.parameters.segment_lengths[0]+self.parameters.segment_lengths[2],
                           self.parameters.segment_lengths[0]+np.cos(self.parameters.joint_angle[0])*self.parameters.segment_lengths[2])
        max_r_length = max(np.cos(self.parameters.joint_angle[1])*self.parameters.segment_lengths[1]+self.parameters.segment_lengths[3],
                           self.parameters.segment_lengths[1]+np.cos(self.parameters.joint_angle[1])*self.parameters.segment_lengths[3])
        self.max_digit_len = max(max_l_length, max_r_length)
        self.MIN_POS = np.array([-self.BASE_RADIUS, FLOOR_Y+self.SEGMENT_OFFSET+self.max_digit_len])
        self.MAX_POS = np.array([WINDOW_SIZE[0]+self.BASE_RADIUS, WINDOW_SIZE[1]+self.BASE_RADIUS+self.max_digit_len])
        self.segment_vertices = [get_rect_vertices((self.SEGMENT_WIDTH, length)) for length in self.parameters.segment_lengths]
        segment_mois = [pymunk.moment_for_poly(self.SEGMENT_MASS, vertices) for vertices in self.segment_vertices]
        
        # Hinge
        self._hinge = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self._hinge.position = position
        self.hinge_shape = pymunk.Circle(self._hinge, self.BASE_RADIUS)
        self.hinge_shape.friction = self.SEGMENT_FRICTION
        self.hinge_shape.filter = self.SEGMENT_COLLISION_FILTER
        space.add(self._hinge, self.hinge_shape)

        # Digit upper segments
        self._segment_ul_body = pymunk.Body(self.SEGMENT_MASS, segment_mois[0], body_type=pymunk.Body.DYNAMIC)
        self._segment_ul_body.position = self._hinge.position + (self.SEGMENT_OFFSET * np.array([-1, -1]))
        self._segment_ul_body.angle = self.MIN_ANGLE
        self._segment_ul = pymunk.Poly(self._segment_ul_body, self.segment_vertices[0])
        self._segment_ul.friction = self.SEGMENT_FRICTION
        self._segment_ul.filter = self.SEGMENT_L_COLLISION_FILTER
        space.add(self._segment_ul_body, self._segment_ul)
        self._segment_ur_body = pymunk.Body(self.SEGMENT_MASS, segment_mois[1], body_type=pymunk.Body.DYNAMIC)
        self._segment_ur_body.position = self._hinge.position + (self.SEGMENT_OFFSET * np.array([1, -1]))
        self._segment_ur_body.angle = -self.MIN_ANGLE
        self._segment_ur = pymunk.Poly(self._segment_ur_body, self.segment_vertices[1])
        self._segment_ur.friction = self.SEGMENT_FRICTION
        self._segment_ur.filter = self.SEGMENT_R_COLLISION_FILTER
        space.add(self._segment_ur_body, self._segment_ur)

        # Digit lower segments
        left_offset = rotate_vertices(np.array([0, self.parameters.segment_lengths[0]]), self._segment_ul_body.angle)
        right_offset = rotate_vertices(np.array([0, self.parameters.segment_lengths[1]]), self._segment_ur_body.angle)
        self._segment_ll_body = pymunk.Body(self.SEGMENT_MASS, segment_mois[2], body_type=pymunk.Body.DYNAMIC)
        self._segment_ll_body.position = self._segment_ul_body.position + left_offset
        self._segment_ll_body.angle = self._segment_ul_body.angle + self.parameters.joint_angle[0]
        self._segment_ll = pymunk.Poly(self._segment_ll_body, self.segment_vertices[2])
        self._segment_ll.friction = self.SEGMENT_FRICTION
        self._segment_ll.filter = self.SEGMENT_L_COLLISION_FILTER
        space.add(self._segment_ll_body, self._segment_ll)
        self._segment_lr_body = pymunk.Body(self.SEGMENT_MASS, segment_mois[3], body_type=pymunk.Body.DYNAMIC)
        self._segment_lr_body.position = self._segment_ur_body.position + right_offset
        self._segment_lr_body.angle = self._segment_ur_body.angle - self.parameters.joint_angle[1]
        self._segment_lr = pymunk.Poly(self._segment_lr_body, self.segment_vertices[3])
        self._segment_lr.friction = self.SEGMENT_FRICTION
        self._segment_lr.filter = self.SEGMENT_R_COLLISION_FILTER
        space.add(self._segment_lr_body, self._segment_lr)

        # Pivot joints make the digits rotate around the hinge
        self._pivot_ul = pymunk.PivotJoint(self._segment_ul_body, self._hinge, (self.SEGMENT_WIDTH/2,0), (-self.SEGMENT_OFFSET,-self.SEGMENT_OFFSET))
        self._pivot_ul.error_bias = 0
        self._pivot_ur = pymunk.PivotJoint(self._segment_ur_body, self._hinge, (self.SEGMENT_WIDTH/2,0), (self.SEGMENT_OFFSET,-self.SEGMENT_OFFSET))
        self._pivot_ur.error_bias = 0
        self._pivot_ll = pymunk.PivotJoint(self._segment_ll_body, self._segment_ul_body, (self.SEGMENT_WIDTH/2,0), (self.SEGMENT_WIDTH/2, self.parameters.segment_lengths[0]))
        self._pivot_ll.error_bias = 0
        self._pivot_lr = pymunk.PivotJoint(self._segment_lr_body, self._segment_ur_body, (self.SEGMENT_WIDTH/2,0), (self.SEGMENT_WIDTH/2, self.parameters.segment_lengths[1]))
        self._pivot_lr.error_bias = 0
        space.add(self._pivot_ul, self._pivot_ur, self._pivot_ll, self._pivot_lr)

        # Rotary joints limit joints to control the opening and closing of the digits
        self._limit_ul = pymunk.RotaryLimitJoint(self._hinge, self._segment_ul_body, self.MIN_ANGLE, self.parameters.rotation_max)
        self._limit_ur = pymunk.RotaryLimitJoint(self._hinge, self._segment_ur_body, -self.parameters.rotation_max, -self.MIN_ANGLE)
        self._limit_ll = pymunk.RotaryLimitJoint(self._segment_ll_body, self._segment_ul_body, -self.parameters.joint_angle[0], -self.parameters.joint_angle[0])
        self._limit_lr = pymunk.RotaryLimitJoint(self._segment_lr_body, self._segment_ur_body, self.parameters.joint_angle[1], self.parameters.joint_angle[1])
        space.add(self._limit_ul, self._limit_ur, self._limit_ll, self._limit_lr)

    def move(self, direction, rotation, open_hand):
        velocity = self.MOVE_SPEED * direction
        self._hinge.position = np.clip(np.array([self._hinge.position])+velocity, self.MIN_POS, self.MAX_POS)[-1].tolist()

        # Apply force to the digits
        force = self.FORCE * open_hand
        force_direction_l = [force, 0]
        force_direction_r = [-force, 0]
        force_position = [self.SEGMENT_WIDTH/2, self.parameters.segment_lengths[0]]
        self._segment_ul_body.apply_force_at_local_point(force_direction_l, force_position)
        self._segment_ur_body.apply_force_at_local_point(force_direction_r, force_position)

    def get_state(self):
        return np.array([self._hinge.position[0]/WINDOW_SIZE[0], 
                        self._hinge.position[1]/WINDOW_SIZE[1]])
 
    def draw(self, canvas):
        pygame.draw.circle(canvas,
                           self.BASE_COLOR,
                           self._hinge.position,
                           self.BASE_RADIUS)
        pygame.draw.polygon(canvas,
                            self.SEGMENT_COLOR,
                            np.array([self._segment_ul_body.position]) + rotate_vertices(self.segment_vertices[0], self._segment_ul_body.angle))
        pygame.draw.polygon(canvas,
                            self.SEGMENT_COLOR,
                            np.array([self._segment_ur_body.position]) + rotate_vertices(self.segment_vertices[1], self._segment_ur_body.angle))
        pygame.draw.polygon(canvas,
                            self.SEGMENT_COLOR,
                            np.array([self._segment_ll_body.position]) + rotate_vertices(self.segment_vertices[2], self._segment_ll_body.angle))
        pygame.draw.polygon(canvas,
                            self.SEGMENT_COLOR,
                            np.array([self._segment_lr_body.position]) + rotate_vertices(self.segment_vertices[3], self._segment_lr_body.angle))

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
        elbow  = 5
        pivot  = 6
        random = 7

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
    CROSS_WIDTH = SIZE/6
    CROSS_MOI = 2*pymunk.moment_for_box(MASS/2, (SIZE, CROSS_WIDTH))
    RANDOM_ANGLE = PI2 / RANDOM_PRECISION


    def __init__(self, space, rng, obj_type):
        # Setup the object
        if obj_type is None:
            self._type = rng.choice(list(self.ObjectTypes)).value
        else:
            self._type = obj_type
        self._type_vec = np.zeros(len(self.ObjectTypes), dtype=float)
        self._type_vec[self._type] = 1
        if self._type == self.ObjectTypes.circle.value:
            self._body = pymunk.Body(1, self.CIRCLE_MOI, body_type=pymunk.Body.DYNAMIC)
            self._shape = pymunk.Circle(self._body, self.SIZE/2)
        elif self._type == self.ObjectTypes.square.value:
            self._body = pymunk.Body(1, self.SQUARE_MOI, body_type=pymunk.Body.DYNAMIC)
            self._shape = pymunk.Poly(self._body, get_rect_vertices((self.SIZE, self.SIZE), offset=(-self.SIZE/2, -self.SIZE/2)))
        elif self._type == self.ObjectTypes.dome.value:
            self._body = pymunk.Body(1, self.DOME_MOI, body_type=pymunk.Body.DYNAMIC)
            self._shape = pymunk.Poly(self._body, self.DOME_VERTICES)
        elif self._type == self.ObjectTypes.sheet.value:
            self._body = pymunk.Body(1, self.SHEET_MOI, body_type=pymunk.Body.DYNAMIC)
            self._shape = pymunk.Poly.create_box(self._body, (self.SIZE*2, self.SHEET_HEIGHT))
        elif self._type == self.ObjectTypes.cross.value:
            self._body = pymunk.Body(1, self.CROSS_MOI, body_type=pymunk.Body.DYNAMIC)
            self._shape = pymunk.Poly.create_box(self._body, (self.SIZE, self.CROSS_WIDTH))
            self._shape2 = pymunk.Poly.create_box(self._body, (self.CROSS_WIDTH, self.SIZE))
        elif self._type == self.ObjectTypes.elbow.value:
            self._body = pymunk.Body(1, self.CROSS_MOI, body_type=pymunk.Body.DYNAMIC)
            _offset = -(self.CROSS_WIDTH+self.SIZE)/2
            self._shape = pymunk.Poly(self._body, get_rect_vertices((self.CROSS_WIDTH, self.SIZE), offset=(_offset, _offset)))
            self._shape2 = pymunk.Poly(self._body, get_rect_vertices((self.SIZE, self.CROSS_WIDTH), offset=(_offset, _offset)))
        elif self._type == self.ObjectTypes.pivot.value: # pivot
            self._body = pymunk.Body(1, self.CROSS_MOI, body_type=pymunk.Body.DYNAMIC)
            self._shape = pymunk.Poly(self._body, get_rect_vertices((self.CROSS_WIDTH, self.SIZE), offset=(-self.CROSS_WIDTH/2, -self.SIZE/4)))
            self._shape2 = pymunk.Poly(self._body, get_rect_vertices((self.SIZE, self.CROSS_WIDTH), offset=(-self.SIZE/2, -self.CROSS_WIDTH/2-self.SIZE/4)))
        elif self._type == self.ObjectTypes.random.value: # random
            self._body = pymunk.Body(1, self.CIRCLE_MOI, body_type=pymunk.Body.DYNAMIC)
            self._shapes = []
            first_rad = rng.uniform(self.SIZE/8, self.SIZE)
            prev_rad = first_rad
            prev_x = prev_rad
            prev_y = 0
            for i in range(1, RANDOM_PRECISION):
                angle = i * self.RANDOM_ANGLE
                radius = rng.uniform(self.SIZE/4, self.SIZE)
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                self._shapes.append(pymunk.Poly(self._body, [(prev_x, prev_y), (x,y), (0, 0)]))
                prev_rad = radius
                prev_x = x
                prev_y = y
            self._shapes.append(pymunk.Poly(self._body, [(prev_x, prev_y), (first_rad,0), (0, 0)]))
        else:
            self._body = pymunk.Body(1, self.CIRCLE_MOI, body_type=pymunk.Body.DYNAMIC)
            self._shape = pymunk.Circle(self._body, 1)

        # Set the object's properties
        self._body.position = (WINDOW_SIZE[0]/2, FLOOR_Y+self.SIZE) #((WINDOW_SIZE[0]-2*self.SPAWN_X_BUFFER)*rng.random(dtype=float) + self.SPAWN_X_BUFFER, FLOOR_Y+self.SIZE/2)
        if self._type != self.ObjectTypes.random.value:
            self._shape.friction = self.FRICTION
            self._shape.filter = pymunk.ShapeFilter(categories=0b10, mask=pymunk.ShapeFilter.ALL_MASKS())

            if self._type in [self.ObjectTypes.cross.value, self.ObjectTypes.elbow.value, self.ObjectTypes.pivot.value]:
                self._shape2.filter = pymunk.ShapeFilter(categories=0b10, mask=pymunk.ShapeFilter.ALL_MASKS())
                self._shape2.friction = self.FRICTION
                space.add(self._body, self._shape, self._shape2)
            else:
                space.add(self._body, self._shape)

        else:
            for shape in self._shapes:
                shape.friction = self.FRICTION
                shape.filter = pymunk.ShapeFilter(categories=0b10, mask=pymunk.ShapeFilter.ALL_MASKS())
            space.add(self._body, *self._shapes)

    def get_state(self):
        return np.array([self._body.position[0]/WINDOW_SIZE[0], 
                        self._body.position[1]/WINDOW_SIZE[1], 
                        self._body.angle/PI2])
    
    def get_type(self):
        return self._type_vec
    
    def draw(self, canvas):
        if self._type == self.ObjectTypes.circle.value:
            pygame.draw.circle(canvas,
                               self.COLOR,
                               self._body.position,
                               self.SIZE/2)
        elif self._type in [self.ObjectTypes.square.value, self.ObjectTypes.dome.value, self.ObjectTypes.sheet.value]:
            pygame.draw.polygon(canvas,
                                self.COLOR,
                                np.array([self._body.position]) + rotate_vertices(self._shape.get_vertices(), self._body.angle))
        elif self._type == self.ObjectTypes.random.value:
            for shape in self._shapes:
                pygame.draw.polygon(canvas,
                                    self.COLOR,
                                    np.array([self._body.position]) + rotate_vertices(shape.get_vertices(), self._body.angle))
        elif self._type != -1: # Don't draw the -1 type
            pygame.draw.polygon(canvas,
                                self.COLOR,
                                np.array([self._body.position]) + rotate_vertices(self._shape.get_vertices(), self._body.angle))
            pygame.draw.polygon(canvas,
                                self.COLOR,
                                np.array([self._body.position]) + rotate_vertices(self._shape2.get_vertices(), self._body.angle))


class Grasp2DEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    GOAL_RADIUS = 32
    GOAL_ROTATION = 0.1
    GRAVITY = -256
    PHYSICS_TIMESTEP = 1/60
    TARGET_Y_MAX_BUFFER = 96
    TARGET_Y_MIN_BUFFER = 48
    MAX_TIME = 150

    AGENT_SPACE = 2 # x, y
    OBJECT_SPACE = 3 # x, y, angle
    OBS_SPACE = AGENT_SPACE + OBJECT_SPACE

    MOVEMENT_SPACE = 5 # up, down, left, right, none
    ROTATION_SPACE = 3 # clockwise, counterclockwise, none
    OPEN_SPACE = 3 # open, close, none
    ACTION_SPACE = MOVEMENT_SPACE + ROTATION_SPACE + OPEN_SPACE


    def __init__(self, render_mode=None):
        super().__init__()
        self._render_flip = True
        
        # What the agent sees
        self.observation_space = spaces.Box(-1, 1, shape=(self.OBS_SPACE,), dtype=float)

        # What the agent can do
        self.action_space = spaces.MultiDiscrete([self.MOVEMENT_SPACE, self.ROTATION_SPACE, self.OPEN_SPACE])

        # Only needs to be set when changing the hand parameters
        self.hand_parameters = None

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

        self._action_to_opening = {
            HandActions.none.value: 0,
            HandActions.open.value: 1,
            HandActions.close.value: -1
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
        hand_state = self._hand.get_state()
        object_state = self._object.get_state()
        object_state[:2] = object_state[:2] - hand_state[:2]  # Make the object position relative to the hand
        return np.concatenate((hand_state, object_state), dtype=float)
    
    def _get_info(self):
        return {"task_type": self._object.get_type().argmax()}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Get options
        if options is None:
            options = {}
        hand_parameters = options.get("hand_parameters", None)
        if hand_parameters is not None:
            self.hand_parameters = hand_parameters
        object_type = options.get("object_type", None)
        
        self._photo_mode = options.get("photo_mode", False)
            
        self._elapsed_steps = 0

        self._space = pymunk.Space()
        self._space.gravity = (0, self.GRAVITY)
        self._space.collision_slop = 0.5

        # Add the objects to the physical space
        self._floor = Floor(self._space)
        self._hand = Hand(self._space, self.np_random, parameters=self.hand_parameters, centered=self._photo_mode)
        self._object = Object(self._space, self.np_random, obj_type=object_type)

        self._target_position = np.array([((WINDOW_SIZE[0] - 2*self._object.SIZE)*self.np_random.random(dtype=float)) + self._object.SIZE, 
                                          ((WINDOW_SIZE[1] - 2*self._object.SIZE - self.TARGET_Y_MAX_BUFFER)*self.np_random.random(dtype=float)) + self._object.SIZE + self.TARGET_Y_MIN_BUFFER,
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
        self._hand.move(self._action_to_direction[action[0]], self._action_to_rotation[action[1]], self._action_to_opening[action[2]])
        self._space.step(self.PHYSICS_TIMESTEP)

        # Check if terminated
        obj_outofbounds = self._object._body.position[0] < 0 or self._object._body.position[0] > WINDOW_SIZE[0] or self._object._body.position[1] > WINDOW_SIZE[1]
        timeout = self._elapsed_steps >= self.MAX_TIME
        truncated = obj_outofbounds or timeout

        obs = self._get_obs()
        info = self._get_info()

        # Use more iterative rewards to enable better exploration
        obj_pos = obs[self.AGENT_SPACE:self.AGENT_SPACE+2]
        dist_to_obj = np.sqrt(obj_pos[0]**2 + (obj_pos[1]*XY_RATIO)**2) / SQ2
        self._obj_off_ground = self._object._body.position[1] > (self._object.SIZE + FLOOR_Y)
        reward = (1-dist_to_obj)**3 + self._obj_off_ground

        if self.render_mode == "human":
            self._render_frame()
        return obs, reward, False, truncated, info

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

        inv_canvas = pygame.transform.flip(canvas, False, True) # Invert Y
        if self.render_mode == "human":
            self.window.blit(inv_canvas, inv_canvas.get_rect())
            if self._render_flip:
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