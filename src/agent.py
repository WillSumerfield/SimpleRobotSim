"""
My implementation of a PPO agent trained on the Grasper environment.
"""

import os
import glob

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback, CheckpointCallback
import numpy as np

from Grasper.wrappers import BetterExploration


class Agent():
    ENTROPY = 0.1

    def __init__(self, env_id):
        self.env_id = env_id
        self.envs = make_vec_env(lambda: self.make_env(), n_envs=os.cpu_count()-2, vec_env_cls=SubprocVecEnv)

    def make_env(self, render_mode=None):
        if render_mode is None:
            env = gym.make(self.env_id)
        else:
            env = gym.make(self.env_id, render_mode=render_mode)
        env = BetterExploration(env)
        env = gym.wrappers.FlattenObservation(env)
        env = Monitor(env)
        check_env(env)
        return env

    def train(self, continue_training=False, total_timesteps=1e8):
        rew_callback = StopTrainingOnRewardThreshold(reward_threshold=1000, verbose=0)
        eval_callback = EvalCallback(
            self.envs, 
            callback_on_new_best=rew_callback,  # Check stopping condition when new best reward is reached
            eval_freq=1e5,  # Evaluate every 10,000 steps
            n_eval_episodes=25,  # Average over 5 episodes
            deterministic=True
        )
        checkpoint_callback = CheckpointCallback(save_freq=1e4, save_path="./checkpoints", name_prefix="ppo_grasper")

        if continue_training:
            # Find the latest checkpoint
            checkpoint_files = glob.glob("./checkpoints/ppo_grasper_*.zip")
            if not checkpoint_files:
                raise FileNotFoundError("No saved model or checkpoints found.")
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            model = PPO.load(latest_checkpoint, env=self.envs, verbose=1, device="cpu", ent_coef=self.ENTROPY)
        else:
            model = PPO("MlpPolicy", self.envs, verbose=1, device="cpu", ent_coef=self.ENTROPY)

        model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])
        model.save("ppo_grasper")

    def test(self, n_eval_episodes=5):
        # Find latest checkpoint
        model_path = "ppo_grasper.zip"
        if not os.path.exists(model_path):
            # Find the latest checkpoint
            checkpoint_files = glob.glob("./checkpoints/ppo_grasper_*.zip")
            if not checkpoint_files:
                raise FileNotFoundError("No saved model or checkpoints found.")
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            model_path = latest_checkpoint

        # Load model
        print(f"Loading model from {model_path}")
        model = PPO.load(model_path, device="cpu")
        mean_reward, std_reward = evaluate_policy(model, self.envs, n_eval_episodes=n_eval_episodes)
        print(f"Mean reward: {mean_reward} +/- {std_reward}")

        # Render the environment
        env = DummyVecEnv([lambda: self.make_env(render_mode="human")])
        obs = env.reset()
        total_reward = 0
        for _ in range(999):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            print(total_reward, end='\r')
            env.render()
            if done:
                total_reward = 0
                obs = env.reset()