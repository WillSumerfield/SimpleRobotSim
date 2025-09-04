import os
import numpy as np
from gymnasium import utils, spaces
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
        "render_fps": 30,
    }
    DEFAULT_CAMERA_CONFIG = {
        "distance": 8.0,
    }

    XML_PATH = os.path.join(os.path.dirname(__file__), "../assets", "grasper.xml")
    RESET_NOISE = 0.01
    REWARD_SIZE = 1.0
    MAX_EPISODE_LENGTH = 150
    FORCE_TESTING_EPISODE = 100


    def __init__(self, **kwargs):

        utils.EzPickle.__init__(self, **kwargs)

        obs_shape = 8 # X,Y Position, X,Y Obj Position, Obj Rotation, Floor+Fingers Obj contact
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self, self.XML_PATH, 2, observation_space=observation_space, camera_name="main", **kwargs
        )

        self._floor_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "floor")
        self._obj_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "object")
        self._left_finger_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "left_finger_lower_geom")
        self._right_finger_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "right_finger_lower_geom")
        self._palm_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "grasper")
        self._obj_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "object")
        self._grav_mag = np.linalg.norm(self.model.opt.gravity)
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

        if not self._force_testing and self._frames >= Grasp2_5DEnv.FORCE_TESTING_EPISODE:
            self._force_testing = True
            self._grasp_distance = np.linalg.norm(observation[[2, 3]])

        # Keep the ball in the same Y plane and remove yz rotation
        self.data.qpos[1] = 0.0
        self.data.qpos[5:7] = 0.0 # y,z rotation
        # Normalize the rotation quaternion
        self.data.qpos[3:7] = self.data.qpos[3:7] / np.linalg.norm(self.data.qpos[3:7])
        self.data.qvel[1] = 0.0
        self.data.qvel[4:6] = 0.0 # y,z rotation

        # Apply gravity compensation to the hand
        hand_mass = self.model.body_subtreemass[self._palm_body_id]
        self.data.xfrc_applied[self._palm_body_id, 2] = hand_mass * self._grav_mag

        # Apply force testing at the end
        if self._force_testing:
            self.data.xfrc_applied[self._obj_body_id, :3] = 10 * self._obj_mass * self._grav_mag * self._force_vec

        obj_dist = np.linalg.norm(observation[[2, 3]]/3) # Reduce distance to [0,1] range-ish
        if not self._force_testing:
            reward = (1-obj_dist)**3 + 2*float(not contacts['floor'])
        else:
            reward = np.min((self._grasp_distance*1.2 - np.linalg.norm(observation[[2, 3]]), 0), 0) # Punish for losing a grip on the object

        self._total_reward += reward

        if self.render_mode == "human":
            self.render()
        return observation, reward, bool(terminated), bool(truncated), {}

    def _get_obs(self, contacts):
        grasper_pos = self.get_body_com("grasper")[[0,2]].copy()
        obj_pos = self.get_body_com("object")[[0,2]].copy()
        relative_pos = obj_pos - grasper_pos
        floor_contact = float(contacts['floor'])
        left_finger_contact = float(contacts['left_finger'])
        right_finger_contact = float(contacts['right_finger'])
        return np.concatenate((grasper_pos, relative_pos, [obj_pos[1], floor_contact, left_finger_contact, right_finger_contact]), dtype=np.float64)

    def reset_model(self):
        self._frames = 0
        self._force_testing = False
        self._obj_mass = self.model.body_mass[self._obj_body_id]
        self._force_vec = np.random.rand(3)-0.5
        self._force_vec[1] = 0.0
        self._force_vec /= np.linalg.norm(self._force_vec)
        qpos = self.init_qpos.copy()
        qpos[7] = self.init_qpos[7] + np.random.uniform(-1, 1)
        qpos[8] = self.init_qpos[8] + np.random.uniform(-0.2, 0.2)
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
            if self.render_mode == "human":
                glfw.set_key_callback(self.mujoco_renderer.viewer.window, lambda *args, **kwargs: None) # Disable key callbacks
                self.mujoco_renderer.viewer.cam.fixedcamid += 1
                self.mujoco_renderer.viewer.cam.type = mj.mjtCamera.mjCAMERA_FIXED
                self.mujoco_renderer.viewer._hide_menu = True
            self._render_init = True
        return ret
