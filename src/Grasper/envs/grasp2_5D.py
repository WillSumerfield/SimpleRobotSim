import os
import numpy as np
import gymnasium as gym
from gymnasium import utils, error, spaces
from gymnasium.envs.mujoco import MujocoEnv
import mujoco as mj
import glfw


class Grasp2_5DEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }
    DEFAULT_CAMERA_CONFIG = {
        "distance": 8.0,
    }

    XML_PATH = os.path.join(os.path.dirname(__file__), "../assets", "grasper.xml")
    RESET_NOISE = 0.01
    REWARD_SIZE = 1.0
    MAX_EPISODE_LENGTH = 75


    def __init__(self, **kwargs):

        utils.EzPickle.__init__(self, **kwargs)

        obs_shape = 8 # X,Y Position, X,Y Obj Position, Obj Rotation, Floor+Fingers Obj contact
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self, self.XML_PATH, obs_shape, observation_space=observation_space, camera_name="main", **kwargs
        )

        self._floor_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "floor")
        self._obj_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "object")
        self._left_finger_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "left_finger_lower_geom")
        self._right_finger_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "right_finger_lower_geom")
        self._frames = 0
        self._render_init = False
        self._total_reward = 0.0

    @property
    def contacts(self):
        """Returns the contacts in the current simulation step."""
        c = {'floor': False, 'left_finger': False, 'right_finger': False}
        for contact in self.data.contact:
            if contact.geom1 != self._obj_geom_id and contact.geom2 != self._obj_geom_id:
                continue

            if contact.geom1 == self._floor_geom_id or contact.geom2 == self._floor_geom_id:
                c['floor'] = True
            elif contact.geom1 == self._left_finger_geom_id or contact.geom2 == self._left_finger_geom_id:
                c['left_finger'] = True
            elif contact.geom1 == self._right_finger_geom_id or contact.geom2 == self._right_finger_geom_id:
                c['right_finger'] = True
        return c

    @property
    def truncated(self):
        return self._frames > self.MAX_EPISODE_LENGTH

    @property
    def terminated(self):
        obj_pos = self.get_body_com("object")[:3].copy()
        return obj_pos[0] < -2 or obj_pos[0] > 2 or \
               obj_pos[2] > 3

    def step(self, action):
        self._frames += 1
        self.do_simulation(action, self.frame_skip)

        contacts = self.contacts

        terminated = self.terminated
        truncated = self.truncated
        observation = self._get_obs(contacts)

        obj_relative_pos = observation[[3, 5]]/3# The X,Z position of the object
        obj_dist = np.linalg.norm(obj_relative_pos)
        reward = (1-obj_dist)**3 + float(not contacts['floor'])*1
        self._total_reward += reward

        if self.render_mode == "human":
            self.render()
        return observation, reward, bool(terminated), bool(truncated), {}

    def _get_obs(self, contacts):
        grasper_pos = self.get_body_com("grasper")[:2].copy()
        obj_pos = self.get_body_com("object")[:3].copy()
        relative_pos = obj_pos[:2] - grasper_pos
        floor_contact = float(contacts['floor'])
        left_finger_contact = float(contacts['left_finger'])
        right_finger_contact = float(contacts['right_finger'])
        return np.concatenate((grasper_pos, relative_pos, [obj_pos[2], floor_contact, left_finger_contact, right_finger_contact]), dtype=np.float64)

    def reset_model(self):
        self._frames = 0
        qpos = self.init_qpos.copy()
        qpos[0] = self.init_qpos[0] + np.random.uniform(-0.1, 0.1)
        qpos[1] = self.init_qpos[1] + np.random.uniform(-0.1, 0.1)
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        
        self._total_reward = 0.0
        self._frames = 0

        contacts = self.contacts

        observation = self._get_obs(contacts)

        return observation
    
    def render(self):
        ret = super().render()
        if not self._render_init:
            glfw.set_key_callback(self.mujoco_renderer.viewer.window, lambda *args, **kwargs: None) # Disable key callbacks
            self.mujoco_renderer.viewer.cam.fixedcamid += 1
            self.mujoco_renderer.viewer.cam.type = mj.mjtCamera.mjCAMERA_FIXED
            self.mujoco_renderer.viewer._hide_menu = True
            self._render_init = True
        
        print(f"Reward: {self._total_reward:.2f}, Frames: {self._frames}        ", end="\r")
        return ret
