"""
My implementation of a PPO agent trained on the Grasper environment.
"""

import os
import glob

import gymnasium as gym
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnRewardThreshold, EvalCallback, CheckpointCallback

from Grasper.wrappers import BetterExploration


ENTROPY = 0.1
CHECKPOINTS_FOLDER = "./checkpoints"
CHECKPOINT_NAME = "ppo_grasper"
VIDEOS_FOLDER = "./videos"
LOGS_FOLDER = "./training_logs"
PPO_ARGS = { "verbose": 1, "device": "cpu", "ent_coef": ENTROPY, "tensorboard_log": LOGS_FOLDER }


def _make_env(env_id, record=False):
    if not record:
        env = gym.make(env_id)
    else:
        env = gym.make(env_id, render_mode="rgb_array")
    env = BetterExploration(env)
    env = gym.wrappers.FlattenObservation(env)
    env = Monitor(env)
    if not record:
        check_env(env)
    else:
        env = gym.wrappers.RecordVideo(env, video_folder=VIDEOS_FOLDER, episode_trigger=lambda x: True, disable_logger=True)
    return env

class SubtaskRewardLogger(BaseCallback):
    def __init__(self, env, verbose=1):
        super().__init__(verbose)
        self.env = env

    def _on_rollout_end(self):
        subtask_rewards_all = [func() for func in self.env.get_attr("get_subtask_performance")]
        # Find the average reward for each subtask across all cores
        subtask_rewards = dict()
        for core in subtask_rewards_all:
            for subtask, reward in core.items():
                if subtask not in subtask_rewards:
                    subtask_rewards[subtask] = 0
                subtask_rewards[subtask] += reward
        for subtask in subtask_rewards:
            subtask_rewards[subtask] /= len(subtask_rewards_all)
        # Log the average reward for each subtask
        for subtask, reward in subtask_rewards.items():
            self.logger.record(f"subtasks/{subtask}_ep_rew_mean", reward)
        return True
    
    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        pass


def train_agent(env_id, continue_training=False, total_timesteps=1e8):
    envs = make_vec_env(lambda: _make_env(env_id), n_envs=os.cpu_count()-2, vec_env_cls=SubprocVecEnv)

    # Callback Functions
    rew_callback = StopTrainingOnRewardThreshold(reward_threshold=1000, verbose=0)
    eval_callback = EvalCallback(
        envs, 
        callback_on_new_best=rew_callback,  # Check stopping condition when new best reward is reached
        eval_freq=1e5,
        n_eval_episodes=25,
        deterministic=True
    )
    checkpoint_callback = CheckpointCallback(save_freq=1e4, save_path=CHECKPOINTS_FOLDER, name_prefix=CHECKPOINT_NAME)
    subtask_reward_callback = SubtaskRewardLogger(envs)
    
    # Load or create model
    if continue_training:
        checkpoint_files = glob.glob(f"{CHECKPOINTS_FOLDER}/{CHECKPOINT_NAME}_*.zip")
        if not checkpoint_files:
            raise FileNotFoundError("No saved model or checkpoints found.")
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        model = PPO.load(latest_checkpoint, envs, **PPO_ARGS)
    else:
        model = PPO("MlpPolicy", envs, **PPO_ARGS)

    # Train model
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback, subtask_reward_callback])
    model.save(CHECKPOINT_NAME)

def test_agent(env_id, n_eval_episodes=5):
    # Find latest checkpoint
    model_path = f"{CHECKPOINT_NAME}.zip"
    if not os.path.exists(model_path):
        # Find the latest checkpoint
        checkpoint_files = glob.glob(f"{CHECKPOINTS_FOLDER}/{CHECKPOINT_NAME}_*.zip")
        if not checkpoint_files:
            raise FileNotFoundError("No saved model or checkpoints found.")
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        model_path = latest_checkpoint

    # Load model
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path, device="cpu")

    # Evaluate model
    env = make_vec_env(lambda: _make_env(env_id), n_envs=1, vec_env_cls=DummyVecEnv)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Render the environment
    gym.logger.min_level = logging.ERROR
    env = _make_env(env_id, record=True)
    obs, info = env.reset()
    for _ in range(999):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, terminated, info = env.step(action)
        env.render()
        if done or terminated:
            break
    env.close()
    print(f"Video saved to {VIDEOS_FOLDER} folder.")
