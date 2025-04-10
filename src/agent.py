"""
My implementation of a PPO agent trained on the Grasper environment.
"""

import os
import glob
import logging
from typing import Optional
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
import gymnasium.spaces as spaces
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnRewardThreshold, EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.policies import ActorCriticPolicy

from Grasper.wrappers import BetterExploration, HandParams
from baseline import load_baseline


CPU_COUNT = os.cpu_count()-4
CHECKPOINTS_FOLDER = "./checkpoints"
CHECKPOINT_NAME = "ppo_grasper"
VIDEOS_FOLDER = "./videos"
LOGS_FOLDER = "./training_logs"
MODEL_FOLDER = "./models"
POLICY_ARGS = {"net_arch": [128, 64], "activation_fn": torch.nn.ReLU}
PPO_ARGS = {"learning_rate": 1e-4, "policy_kwargs": POLICY_ARGS, "verbose": 1, "device": "cpu", "batch_size": 256, 
            "ent_coef": 0.05, "gamma": 0.98}
PARAM_DICT = {"learning_rate": [3e-4, 1e-4], "n_steps": [512, 1024]}


def _make_env(env_id, hand_type, record=False):
    if not record:
        env = gym.make(env_id)
    else:
        env = gym.make(env_id, render_mode="rgb_array")
    env = BetterExploration(env)
    env = HandParams(env, hand_type)
    env = gym.wrappers.FlattenObservation(env)
    env = Monitor(env)
    if not record:
        check_env(env)
    else:
        env = gym.wrappers.RecordVideo(env, video_folder=f"{VIDEOS_FOLDER}/hand_type_{hand_type}", episode_trigger=lambda x: True, disable_logger=True)
    return env


class SubtaskRewardLogger(BaseCallback):
    def __init__(self, env, verbose=1):
        super().__init__(verbose)
        self.env = env

    def _on_rollout_end(self):
        subtask_rewards_all = [func() for func in self.env.get_attr("get_subtask_performance")]
        # Sum the rewards across all cores
        subtask_rewards = dict()
        subtask_count = dict()
        for core in subtask_rewards_all:
            for subtask, reward in core.items():
                if subtask not in subtask_rewards:
                    subtask_rewards[subtask] = 0
                    subtask_count[subtask] = 0
                subtask_rewards[subtask] += reward
                subtask_count[subtask] += 1
        # Average the rewards
        for subtask in subtask_rewards:
            subtask_rewards[subtask] /= subtask_count[subtask]
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


class DAPG(PPO):

    class CustomActorCriticPolicy(ActorCriticPolicy):
        def evaluate_actions(self, obs: PyTorchObs, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            # Preprocess the observation if needed
            features = self.extract_features(obs)
            if self.share_features_extractor:
                latent_pi, latent_vf = self.mlp_extractor(features)
            else:
                pi_features, vf_features = features
                latent_pi = self.mlp_extractor.forward_actor(pi_features)
                latent_vf = self.mlp_extractor.forward_critic(vf_features)
            distribution = self._get_action_dist_from_latent(latent_pi)
            log_prob = distribution.log_prob(actions)
            values = self.value_net(latent_vf)
            entropy = distribution.entropy()
            logits = torch.cat([dist.probs for dist in distribution.distribution], dim=1)
            return values, log_prob, entropy, logits


    def __init__(self, **kwargs):
        args = {key: value for key, value in PPO_ARGS.items() if key not in kwargs}
        if "policy" not in kwargs:
            args["policy"] = self.CustomActorCriticPolicy
        super().__init__(**{**args, **kwargs})
        self.baseline_policy = load_baseline(self.observation_space.shape[0], self.action_space.nvec.tolist(), device=self.device, verbose=self.verbose)
        self.baseline_policy.eval()
        for param in self.baseline_policy.parameters():
            param.requires_grad = False

        self.lambda_bc_init = 0
        self.lambda_bc_decay = 0.99

    # Modified PPO class to include Behavior Cloning
    def train(self) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy, logits = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                rl_loss = policy_loss + self.ent_coef*entropy_loss + self.vf_coef*value_loss

                # Compute Behavior Cloning Loss
                bc_dist = self.baseline_policy(rollout_data.observations)
                bc_loss = F.binary_cross_entropy_with_logits(logits, bc_dist, reduction="mean")
                
                # Decaying Lambda for BC
                lambda_bc = self.lambda_bc_init * (self.lambda_bc_decay ** self._n_updates)
                
                # Total Loss
                loss = lambda_bc*bc_loss + rl_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        self.logger.record("train/lambda_bc", lambda_bc)
        self.logger.record("train/rl_loss", rl_loss.item())
        self.logger.record("train/bc_loss", bc_loss.item()*lambda_bc)


def train_agent(env_id, hand_type, continue_training=False, total_timesteps=1e8, param_dict=dict(), verbose=1, provided_envs=None):
    if provided_envs is None:
        envs = make_vec_env(lambda: _make_env(env_id, hand_type), n_envs=CPU_COUNT, vec_env_cls=SubprocVecEnv)
    else:
        envs = provided_envs

    checkpoint_folder = f"{CHECKPOINTS_FOLDER}/hand_type_{hand_type}"

    # Callback Functions
    rew_callback = StopTrainingOnRewardThreshold(reward_threshold=1100, verbose=verbose)
    eval_callback = EvalCallback(
        envs, 
        callback_on_new_best=rew_callback,  # Check stopping condition when new best reward is reached
        eval_freq=2e4,
        n_eval_episodes=25,
        deterministic=True,
        verbose=verbose
    )
    checkpoint_callback = CheckpointCallback(save_freq=1e4, save_path=checkpoint_folder, name_prefix=CHECKPOINT_NAME, verbose=verbose)
    subtask_reward_callback = SubtaskRewardLogger(envs, verbose=verbose)

    tensorboard_log = f"{LOGS_FOLDER}/hand_type_{hand_type}"
    
    # Load or create model
    if continue_training:
        checkpoint_files = glob.glob(f"{checkpoint_folder}/{CHECKPOINT_NAME}_*.zip")
        if not checkpoint_files:
            raise FileNotFoundError("No saved model or checkpoints found.")
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        ppo_args = PPO_ARGS.copy()
        ppo_args["verbose"] = verbose
        ppo_args["tensorboard_log"] = tensorboard_log
        print(f"Loading model from {latest_checkpoint}.")
        model = DAPG.load(path=latest_checkpoint, env=envs, **ppo_args, kwargs=param_dict)
    else:
        model = DAPG(env=envs, **param_dict, verbose=verbose, tensorboard_log=tensorboard_log)

    # Train model
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback, subtask_reward_callback])
    model.save(f"{MODEL_FOLDER}/hand_type_{hand_type}/{CHECKPOINT_NAME}")
    return model


def test_agent(env_id, hand_type, n_eval_episodes=250, n_displayed_episodes=5):
    # Find latest checkpoint
    checkpoint_files = glob.glob(f"{CHECKPOINTS_FOLDER}/hand_type_{hand_type}/{CHECKPOINT_NAME}_*.zip")
    if not checkpoint_files:
        raise FileNotFoundError("No saved model or checkpoints found.")
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)

    env = make_vec_env(lambda: _make_env(env_id, hand_type), n_envs=1, vec_env_cls=DummyVecEnv)

    # Load model
    print(f"Loading model from {latest_checkpoint}")
    model = DAPG.load(env=env, path=latest_checkpoint, **PPO_ARGS, tensorboard_log=f"{LOGS_FOLDER}/hand_type_{hand_type}")

    # Evaluate model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Render the environment
    gym.logger.min_level = logging.ERROR
    env = _make_env(env_id, hand_type, record=True)
    episodes = 1
    obj_type = 0
    obs, info = env.reset(options={"object_type": obj_type})
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, terminated, info = env.step(action)
        env.render()
        if done or terminated:
            if episodes >= n_displayed_episodes:
                break
            obj_type += 1
            obs, info = env.reset(options={"object_type": obj_type})
            episodes += 1
    env.close()
    print(f"Video saved to {VIDEOS_FOLDER} folder.")


def param_sweep(env_id, hand_type, param_dict=None, timesteps_per_param=1e6):
    envs = make_vec_env(lambda: _make_env(env_id, hand_type), n_envs=CPU_COUNT, vec_env_cls=SubprocVecEnv)
    if param_dict is None:
        param_dict = PARAM_DICT
    param_combinations = get_param_combinations(param_dict)
    best_performance = -float("inf")
    best_params = None
    for idx, params in enumerate(param_combinations):
        print(f"{idx+1}/{len(param_combinations)} Training with {params}", end="")
        model = train_agent(env_id, hand_type, total_timesteps=timesteps_per_param, param_dict=params, verbose=0, provided_envs=envs)
        env = make_vec_env(lambda: _make_env(env_id, hand_type), n_envs=1, vec_env_cls=DummyVecEnv)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=25, deterministic=True)
        del model
        if mean_reward > best_performance:
            best_performance = mean_reward
            best_params = params
        print(f" - Achieved {mean_reward}.")
    print(f"Best performance {best_performance} found using {best_params}!")


def get_param_combinations(param_dict):
    param_names = list(param_dict.keys())
    param_values = [param_dict[name] for name in param_names]
    value_combinations = list(product(*param_values))
    param_combinations = [dict(zip(param_names, values)) for values in value_combinations]
    return param_combinations