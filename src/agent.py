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
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback, CheckpointCallback

from Grasper.wrappers import BetterExploration


ENTROPY = 0.1


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
        env = gym.wrappers.RecordVideo(env, video_folder='./videos', 
                                       episode_trigger=lambda x: True, disable_logger=True)
    return env

def train_agent(env_id, continue_training=False, total_timesteps=1e8):
    envs = make_vec_env(lambda: _make_env(env_id), n_envs=2, vec_env_cls=SubprocVecEnv)

    # Callback Functions
    rew_callback = StopTrainingOnRewardThreshold(reward_threshold=1000, verbose=0)
    eval_callback = EvalCallback(
        envs, 
        callback_on_new_best=rew_callback,  # Check stopping condition when new best reward is reached
        eval_freq=1e5,  # Evaluate every 10,000 steps
        n_eval_episodes=25,  # Average over 5 episodes
        deterministic=True
    )
    checkpoint_callback = CheckpointCallback(save_freq=1e4, save_path="./checkpoints", name_prefix="ppo_grasper")
    
    # Load or create model
    if continue_training:
        checkpoint_files = glob.glob("./checkpoints/ppo_grasper_*.zip")
        if not checkpoint_files:
            raise FileNotFoundError("No saved model or checkpoints found.")
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        model = PPO.load(latest_checkpoint, env=envs, verbose=1, device="cpu", ent_coef=ENTROPY)
    else:
        model = PPO("MlpPolicy", envs, verbose=1, device="cpu", ent_coef=ENTROPY)

    # Train model
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])
    model.save("ppo_grasper")

def test_agent(env_id, n_eval_episodes=5):
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

    # Evaluate model
    env = make_vec_env(lambda: _make_env(env_id), n_envs=1, vec_env_cls=DummyVecEnv)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Render the environment
    gym.logger.min_level = logging.ERROR
    env = _make_env(env_id, record=True)
    obs, info = env.reset()
    for _ in range(999):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, terminated, info = env.step(action)
        env.render()
        if done:
            break
    env.close()
    print(f"Video saved to 'videos' folder.")
