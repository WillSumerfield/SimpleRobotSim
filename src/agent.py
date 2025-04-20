"""
My implementation of a PPO agent trained on the Grasper environment.
"""

import os
import glob
import logging
from typing import Optional, Callable, Any
from itertools import product
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnRewardThreshold, EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.buffers import RolloutBuffer

from Grasper.wrappers import BetterExploration, HandParams, TaskType
from baseline import BASELINE_PATH, load_baseline


CPU_COUNT = os.cpu_count()-4
REWARD_THRESHOLD = 1000
CHECKPOINTS_FOLDER = "./checkpoints"
CHECKPOINT_NAME = "ppo_grasper"
VIDEOS_FOLDER = "./videos"
LOGS_FOLDER = "./training_logs"
MODEL_FOLDER = "./models"
POLICY_ARGS = {"net_arch": [128, 128], "activation_fn": torch.nn.ReLU, "optimizer_class": torch.optim.Adam}
PPO_ARGS = {"learning_rate": 1e-3, "policy_kwargs": POLICY_ARGS, "verbose": 1, "device": "cpu", "batch_size": 256, 
            "ent_coef": 0.01, "gamma": 0.98, "n_epochs": 5}
PARAM_DICT = {"learning_rate": [3e-4, 1e-4], "n_steps": [512, 1024]}


def _make_env(env_id, hand_type, task_type, record=False):
    if not record:
        env = gym.make(env_id)
    else:
        env = gym.make(env_id, render_mode="rgb_array")
    env = BetterExploration(env)
    env = HandParams(env, hand_type)
    if task_type is not None:
        env = TaskType(env, task_type)
    env = gym.wrappers.FlattenObservation(env)
    env = Monitor(env)
    if not record:
        check_env(env)
    else:
        if task_type is None:
            video_folder = f"{VIDEOS_FOLDER}/hand_type_{hand_type}"
        else:
            video_folder = f"{VIDEOS_FOLDER}/hand_type_{hand_type}_task_{task_type}"
        env = gym.wrappers.RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True, disable_logger=True)
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


class PopArt(nn.Module):
    def __init__(self, input_dim, beta=0.99, epsilon=1e-5):
        """
        Initializes a PopArt layer.
        
        Args:
            input_dim (int): Dimension of the input from the encoder.
            beta (float): EMA decay factor.
            epsilon (float): Small value to ensure numerical stability.
        """
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
        
        # Instead of using nn.Parameter (with gradients disabled),
        # we register these as buffers so they move with the model.
        self.register_buffer('mu', torch.tensor(0.0))
        self.register_buffer('sigma', torch.tensor(1.0))
        self.register_buffer('mean_square', torch.tensor(1.0))
        
        # The final linear layer predicts a normalized value.
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        """
        Returns the unnormalized value by scaling the normalized prediction.
        """
        # The network outputs a normalized value. We then rescale it using the current sigma and mu.
        return self.sigma * self.linear(x) + self.mu

    def update(self, targets):
        """
        Updates the running estimates for mu and sigma using an exponential moving average.
        It then adjusts the weights of the linear layer so that the unnormalized output remains unchanged.
        
        Args:
            targets (Tensor): The target returns as a 1D tensor.
        """
        with torch.no_grad():
            # Save the old values for the adjustment.
            old_mu = self.mu.clone()
            old_sigma = self.sigma.clone()
            
            # Compute statistics from the batch of target returns.
            batch_mean = targets.mean()
            batch_mean_sq = (targets**2).mean()
            
            # Update running averages with EMA.
            new_mean = self.beta * self.mu + (1 - self.beta) * batch_mean
            new_mean_sq = self.beta * self.mean_square + (1 - self.beta) * batch_mean_sq
            new_sigma = torch.sqrt(new_mean_sq - new_mean**2 + self.epsilon)
            
            # Update buffers.
            self.mu.copy_(new_mean)
            self.mean_square.copy_(new_mean_sq)
            self.sigma.copy_(new_sigma)
            
            # Adjust the output layer's parameters to preserve the unnormalized value.
            # Let f(x) = W*x + b, and our value prediction is V(x)=sigma*f(x) + mu.
            # When mu and sigma change, we update:
            #   W' = (old_sigma / new_sigma) * W
            #   b' = (old_sigma * b + old_mu - new_mean) / new_sigma
            self.linear.weight.data.mul_(old_sigma / new_sigma)
            self.linear.bias.data.mul_(old_sigma / new_sigma)
            self.linear.bias.data.add_((old_mu - new_mean) / new_sigma)


class MultiTaskMlpPolicy(ActorCriticPolicy):

    def __init__(self, *args, **kwargs):
        self.task_count = kwargs.pop("task_count", 1)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_nets = nn.ModuleList([self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi) for _ in range(self.task_count)])
        self.value_nets = nn.ModuleList([nn.Linear(self.mlp_extractor.latent_dim_vf, 1) for _ in range(self.task_count)])

        # Init weights: use orthogonal initialization with small initial weight for the output
        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                "action_nets": (self.action_nets, 0.01),
                "value_nets": (self.value_nets, 1),
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                if isinstance(module, str):
                    for sub_module in gain[0]:
                        sub_module.apply(partial(self.init_weights, gain=gain[1]))
                else:
                    module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_task_id(self, obs: PyTorchObs) -> int:
        return obs[:, 0:self.task_count].argmax(dim=1)

    def _get_action_dist_from_latent(self, task_id: int, latent_pi: torch.Tensor) -> Distribution:
        mean_actions = self.action_nets[task_id](latent_pi)
        return self.action_dist.proba_distribution(action_logits=mean_actions)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # Evaluate the values for the given observations
        task_ids = self._get_task_id(obs)
        actions = torch.empty((obs.shape[0], self.action_space.nvec.shape[0]), dtype=torch.int64, device=obs.device)
        values = torch.empty(obs.shape[0], dtype=torch.float32, device=obs.device)
        log_prob = torch.empty(obs.shape[0], dtype=torch.float32, device=obs.device)
        for task_id in range(self.task_count):
            task_indices = (task_ids == task_id).nonzero(as_tuple=True)[0]
            if len(task_indices) == 0:
                continue
            distribution = self._get_action_dist_from_latent(task_id, latent_pi[task_indices])
            actions[task_indices] = distribution.get_actions(deterministic=deterministic)
            values[task_indices] = self.value_nets[task_id](latent_vf[task_indices]).flatten()
            log_prob[task_indices] = distribution.log_prob(actions[task_indices])
        
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def evaluate_actions(self, task_id: int, obs: PyTorchObs, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(task_id, latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_nets[task_id](latent_vf)
        entropy = distribution.entropy()
        logits = torch.cat([dist.probs for dist in distribution.distribution], dim=1)
        return values, log_prob, entropy, logits
    
    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        task_ids = self._get_task_id(obs)
        logits = [torch.empty((obs.shape[0], space), dtype=torch.float32, device=obs.device) for space in self.action_space.nvec]
        for task_id in range(self.task_count):
            task_indices = (task_ids == task_id).nonzero(as_tuple=True)[0]
            if len(task_indices) == 0:
                continue
            dist = self._get_action_dist_from_latent(task_id, latent_pi[task_indices])
            for i in range(self.action_space.nvec.shape[0]):
                logits[i][task_indices] = dist.distribution[i].logits
        # Combine the distributions for all tasks
        dists = self.action_dist.proba_distribution(action_logits=torch.cat(logits, dim=1))
        return dists

    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        task_ids = self._get_task_id(obs)
        values = torch.empty(obs.shape[0], dtype=torch.float32, device=obs.device)
        for task_id in range(self.task_count):
            task_indices = (task_ids == task_id).nonzero(as_tuple=True)[0]
            if len(task_indices) == 0:
                continue
            values[task_indices] = self.value_nets[task_id](latent_vf[task_indices]).flatten()
        return values


class DAPG(PPO):
    def __init__(self, **kwargs):
        args = {key: value for key, value in PPO_ARGS.items() if key not in kwargs}
        if "policy" not in kwargs:
            args["policy"] = MultiTaskMlpPolicy
        args["policy_kwargs"]["task_count"] = kwargs["env"].get_attr("OBJECT_TYPES")[0]
        super().__init__(**{**args, **kwargs})
        self.baseline_policy = load_baseline(device=self.device, verbose=self.verbose)
        self.baseline_policy.eval()
        for param in self.baseline_policy.parameters():
            param.requires_grad = False

        self.lambda_bc_init = 0.3
        self.lambda_bc_decay = 0.98
        self.task_count = self.env.get_attr("OBJECT_TYPES")[0]
        self.subtask_distribution = None

        # Normalize value for each reward
        self.beta = 0.999
        self.epsilon = 1e-5
        self.subtask_value_mu = [torch.tensor(0.0) for _ in range(self.task_count)]
        self.subtask_value_sigma = [torch.tensor(1.0) for _ in range(self.task_count)]
        self.subtask_value_mean_square = [torch.tensor(1.0) for _ in range(self.task_count)]

    def _get_task_id(self, obs: PyTorchObs) -> int:
        return obs[:, 0:self.task_count].argmax(dim=1)

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
        rl_losses, bc_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions

                # Get the loss per task
                task_ids = self._get_task_id(rollout_data.observations)
                for task_id in range(self.task_count):
                    task_indices = (task_ids == task_id).nonzero(as_tuple=True)[0]
                    if len(task_indices) == 0:
                        continue
                    obs = rollout_data.observations[task_indices]
                    acts = actions[task_indices]
                    values, log_prob, entropy, logits = self.policy.evaluate_actions(task_id, obs, acts)
                    values = values.flatten()

                    # Normalize advantage
                    advantages = rollout_data.advantages[task_indices]
                    # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # ratio between old and new policy, should be one at the first iteration
                    ratio = torch.exp(log_prob - rollout_data.old_log_prob[task_indices])

                    # clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                    # Logging
                    pg_losses.append(policy_loss.item())
                    clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                    clip_fractions.append(clip_fraction)

                    # Value function loss
                    if self.clip_range_vf is None:
                        # No clipping
                        values_pred = values
                    else:
                        # Clip the difference between old and new value
                        # NOTE: this depends on the reward scaling
                        values_pred = rollout_data.old_values[task_indices] + torch.clamp(
                            values - rollout_data.old_values[task_indices], -clip_range_vf, clip_range_vf
                        )
                    # Value loss using the TD(gae_lambda) target
                    target_values = rollout_data.returns[task_indices]
                    value_loss = F.mse_loss(target_values, values_pred)
                    value_loss = self._normalize_subtask_value_loss(value_loss, task_id)
                    value_losses.append(value_loss.item())

                    # Entropy loss favor exploration
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = -torch.mean(-log_prob)
                    else:
                        entropy_loss = -torch.mean(entropy)

                    entropy_losses.append(entropy_loss.item())

                    rl_loss = policy_loss + self.ent_coef*entropy_loss + self.vf_coef*value_loss
                    rl_losses.append(rl_loss.item())

                    # Compute Behavior Cloning Loss
                    bc_dist = self.baseline_policy(rollout_data.observations[task_indices])
                    bc_loss = F.binary_cross_entropy_with_logits(logits, bc_dist, reduction="mean")
                    bc_losses.append(bc_loss.item())
                    
                    # Decaying Lambda for BC
                    lambda_bc = self.lambda_bc_init * (self.lambda_bc_decay ** self._n_updates)
                    
                    # Total Loss
                    loss = lambda_bc*bc_loss + rl_loss

                    # Calculate approximate form of reverse KL Divergence for early stopping
                    # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                    # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                    # and Schulman blog: http://joschu.net/blog/kl-approx.html
                    with torch.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob[task_indices]
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
        self.logger.record("train/rl_loss", np.mean(rl_losses))
        self.logger.record("train/bc_loss", np.mean(bc_losses)*lambda_bc)
        if self.subtask_distribution is not None:
            for i in range(self.task_count):
                self.logger.record(f"train/subtask_distribution_{i}", self.subtask_distribution[i])
                self.logger.record(f"train/subtask_counts_{i}", self.subtask_counts[i])

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        # Update the ratio of subtasks based on the most recent rewards
        # subtask_rewards = [0]*self.task_count
        # self.subtask_counts = [0]*self.task_count
        # for ep in self.ep_info_buffer:
        #     task = ep["subtask_type"]
        #     subtask_rewards[task] += ep["r"]
        #     self.subtask_counts[task] += 1
        # subtask_sigmoid = [1/(1+np.exp(reward/200)) for reward in subtask_rewards]
        # subtask_sum = sum(subtask_sigmoid)
        # self.subtask_distribution = [ss/subtask_sum for ss in subtask_sigmoid]
        # self.env.set_options({"subtask_distribution": self.subtask_distribution})

        return True
    
    def _update_info_buffer(self, infos: list[dict[str, Any]], dones: Optional[np.ndarray] = None) -> None:
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            if maybe_ep_info is not None:
                subtask_type = info.get("task_type")
                ep_info = {"subtask_type": subtask_type, "r": maybe_ep_info["r"], "l": maybe_ep_info["l"]}
                self.ep_info_buffer.extend([ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)

    def _normalize_subtask_value_loss(self, value_loss: torch.Tensor, subtask_id: int):
        value_loss_nograd = value_loss.detach().clone()
        batch_mean = value_loss_nograd.mean()
        batch_mean_sq = (value_loss_nograd**2).mean()
        
        # Update running averages with EMA
        new_mean = self.beta * self.subtask_value_mu[subtask_id] + (1 - self.beta) * batch_mean
        new_mean_sq = self.beta * self.subtask_value_mean_square[subtask_id] + (1 - self.beta) * batch_mean_sq
        new_sigma = torch.sqrt(new_mean_sq - new_mean**2 + self.epsilon)
        
        # Update running normalization values
        self.subtask_value_mu[subtask_id].copy_(new_mean)
        self.subtask_value_mean_square[subtask_id].copy_(new_mean_sq)
        self.subtask_value_sigma[subtask_id].copy_(new_sigma)

        return (value_loss - self.subtask_value_mu[subtask_id]) / self.subtask_value_sigma[subtask_id]


def train_agent(env_id, hand_type, task_type, continue_training=False, total_timesteps=2e7, param_dict=dict(), verbose=1, provided_envs=None):
    if provided_envs is None:
        envs = make_vec_env(lambda: _make_env(env_id, hand_type, task_type), n_envs=CPU_COUNT, vec_env_cls=SubprocVecEnv)
    else:
        envs = provided_envs

    checkpoint_folder = f"{CHECKPOINTS_FOLDER}/hand_type_{hand_type}_task_{task_type}"

    # Callback Functions
    rew_callback = StopTrainingOnRewardThreshold(reward_threshold=REWARD_THRESHOLD, verbose=verbose)
    eval_callback = EvalCallback(
        envs, 
        callback_on_new_best=rew_callback,  # Check stopping condition when new best reward is reached
        eval_freq=1e4,
        n_eval_episodes=25,
        deterministic=True,
        verbose=verbose
    )
    checkpoint_callback = CheckpointCallback(save_freq=1e4, save_path=checkpoint_folder, name_prefix=CHECKPOINT_NAME, verbose=verbose)
    subtask_reward_callback = SubtaskRewardLogger(envs, verbose=verbose)

    tensorboard_log = f"{LOGS_FOLDER}/hand_type_{hand_type}_task_{task_type}"
    
    # Load or create model
    if continue_training:
        checkpoint_files = glob.glob(f"{checkpoint_folder}/{CHECKPOINT_NAME}_*.zip")
        if not checkpoint_files:
            raise FileNotFoundError("No saved model or checkpoints found.")
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        ppo_args = PPO_ARGS.copy()
        ppo_args["verbose"] = verbose
        ppo_args["tensorboard_log"] = tensorboard_log
        ppo_args["policy_kwargs"]["task_count"] = envs.get_attr("OBJECT_TYPES")[0]
        print(f"Loading model from {latest_checkpoint}.")
        model = DAPG.load(path=latest_checkpoint, env=envs, **ppo_args, kwargs=param_dict)
    else:
        model = DAPG(env=envs, **param_dict, verbose=verbose, tensorboard_log=tensorboard_log)

    # Train model
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback, subtask_reward_callback])
    model.save(f"{MODEL_FOLDER}/hand_type_{hand_type}_task_{task_type}/{CHECKPOINT_NAME}")
    return model


def test_agent(env_id, hand_type, task_type, n_eval_episodes=100, n_displayed_episodes=5, checkpoint=True):

    # Find latest checkpoint
    if checkpoint:
        if task_type is None:
            checkpoint_files = glob.glob(f"{CHECKPOINTS_FOLDER}/hand_type_{hand_type}/{CHECKPOINT_NAME}_*.zip")
        else:
            checkpoint_files = glob.glob(f"{CHECKPOINTS_FOLDER}/hand_type_{hand_type}_task_{task_type}/{CHECKPOINT_NAME}_*.zip")
        if not checkpoint_files:
            raise FileNotFoundError("No saved model or checkpoints found.")
        model_path = max(checkpoint_files, key=os.path.getctime)

    # Find the specific model
    else:
        if task_type is None:
            model_path = f"{MODEL_FOLDER}/hand_type_{hand_type}/{CHECKPOINT_NAME}"
        else:
            model_path = f"{MODEL_FOLDER}/hand_type_{hand_type}_task_{task_type}/{CHECKPOINT_NAME}"

    env = make_vec_env(lambda: _make_env(env_id, hand_type, task_type), n_envs=1, vec_env_cls=DummyVecEnv)

    # Load model
    print(f"Loading model from {model_path}")
    ppo_args = PPO_ARGS.copy()
    ppo_args["policy_kwargs"]["task_count"] = env.get_attr("OBJECT_TYPES")[0]
    model = DAPG.load(env=env, path=model_path, **ppo_args, tensorboard_log=f"{LOGS_FOLDER}/hand_type_{hand_type}")

    # Evaluate model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Render the environment
    gym.logger.min_level = logging.ERROR
    env = _make_env(env_id, hand_type, task_type, record=True)
    episodes = 1
    obj_type = 0 if task_type is None else task_type
    obs, info = env.reset(options={"object_type": obj_type})
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, terminated, info = env.step(action)
        env.render()
        if done or terminated:
            if episodes >= n_displayed_episodes:
                break
            if task_type is not None:
                break
            obj_type += 1
            obs, info = env.reset(options={"object_type": obj_type})
            episodes += 1
    env.close()
    print(f"Video saved to {VIDEOS_FOLDER} folder.")


def param_sweep(env_id, hand_type, task_type, param_dict=None, timesteps_per_param=1e6):
    envs = make_vec_env(lambda: _make_env(env_id, hand_type, task_type), n_envs=CPU_COUNT, vec_env_cls=SubprocVecEnv)
    if param_dict is None:
        param_dict = PARAM_DICT
    param_combinations = get_param_combinations(param_dict)
    best_performance = -float("inf")
    best_params = None
    for idx, params in enumerate(param_combinations):
        print(f"{idx+1}/{len(param_combinations)} Training with {params}", end="")
        model = train_agent(env_id, hand_type, total_timesteps=timesteps_per_param, param_dict=params, verbose=0, provided_envs=envs)
        env = make_vec_env(lambda: _make_env(env_id, hand_type, task_type), n_envs=1, vec_env_cls=DummyVecEnv)
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


def convert_to_baseline(env_id, hand_type, task_type, checkpoint=True):

    # Find latest checkpoint
    if checkpoint:
        if task_type is None:
            checkpoint_files = glob.glob(f"{CHECKPOINTS_FOLDER}/hand_type_{hand_type}/{CHECKPOINT_NAME}_*.zip")
        else:
            checkpoint_files = glob.glob(f"{CHECKPOINTS_FOLDER}/hand_type_{hand_type}_task_{task_type}/{CHECKPOINT_NAME}_*.zip")
        if not checkpoint_files:
            raise FileNotFoundError("No saved model or checkpoints found.")
        model_path = max(checkpoint_files, key=os.path.getctime)

    # Find the specific model
    else:
        if task_type is None:
            model_path = f"{MODEL_FOLDER}/hand_type_{hand_type}/{CHECKPOINT_NAME}"
        else:
            model_path = f"{MODEL_FOLDER}/hand_type_{hand_type}_task_{task_type}/{CHECKPOINT_NAME}"
    
    env = make_vec_env(lambda: _make_env(env_id, hand_type, task_type), n_envs=1, vec_env_cls=DummyVecEnv)

    # Load model
    print(f"Loading model from {model_path}")
    ppo_args = PPO_ARGS.copy()
    ppo_args["policy_kwargs"]["task_count"] = env.get_attr("OBJECT_TYPES")[0]
    model = DAPG.load(env=env, path=model_path, **ppo_args, tensorboard_log=f"{LOGS_FOLDER}/hand_type_{hand_type}")

    # Combine the latent compressor and the task head
    model = nn.Sequential(model.policy.mlp_extractor.policy_net, model.policy.action_nets[task_type])

    # Save the model
    torch.save(model, f"{BASELINE_PATH}")
    print(f"Baseline model saved to {BASELINE_PATH}")