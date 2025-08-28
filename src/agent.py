"""
My implementation of a PPO agent trained on the Grasper environment.
"""

import os
import glob
import logging
from itertools import product
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
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
from stable_baselines3.common.buffers import RolloutBuffer

from src import *
from src.Grasper.wrappers import HandParams, TaskType
from src.hand_morphologies import unnorm_hand_params


CPU_COUNT = int(os.cpu_count()/2)
REWARD_THRESHOLD = 130
PHOTO_FRAME = 15
TRAIN_EPSISODES = 1e6
EVAL_EPISODES = 500

POLICY_ARGS = {"net_arch": [64, 64], "activation_fn": torch.nn.ReLU, "optimizer_class": torch.optim.Adam}
PPO_ARGS = {"learning_rate": 3e-4, "policy_kwargs": POLICY_ARGS, "verbose": 1, "device": "cpu", "batch_size": 256, 
            "ent_coef": 0.01, "gamma": 0.98, "n_epochs": 10}
PARAM_DICT = {"gamma": [0.98], "ent_coef": [0.01, 0.1]}


def _make_env(env_id, hand_type, task_type, ga_index=None, pt_index=None, render=False, record=False):

    env_folder = ENV_FOLDER_2D if env_id == ENV_2D else ENV_FOLDER_2_5D

    if not render:
        env = gym.make(env_id)
    else:
        env = gym.make(env_id, render_mode="rgb_array")
    if ga_index is None and pt_index is None:
        env = HandParams(env, hand_type)
    env = TaskType(env, task_type)
    env = gym.wrappers.FlattenObservation(env)
    env = Monitor(env)
    if not render:
        check_env(env)
    elif record:
        if ga_index is not None:
            video_folder = f"{VIDEOS_FOLDER}/{env_folder}/{GA_FOLDER}/"
        elif pt_index is not None:
            video_folder = f"{VIDEOS_FOLDER}/{env_folder}/{PT_FOLDER}/task{task_type}/hand{hand_type}"
        else:
            if task_type is None:
                video_folder = f"{VIDEOS_FOLDER}/{env_folder}/hand{hand_type}"
            else:
                video_folder = f"{VIDEOS_FOLDER}/{env_folder}/hand{hand_type}/task{task_type}"
        env = gym.wrappers.RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True, disable_logger=True)
    return env

class DAPG(PPO):
    def __init__(self, **kwargs):
        args = {key: value for key, value in PPO_ARGS.items() if key not in kwargs}
        if "policy" not in kwargs:
            args["policy"] = "MlpPolicy"
        if "verbose" not in kwargs:
            args["verbose"] = 0 # Stupid fix to bug in stable_baselines3
        super().__init__(**{**args, **kwargs})
        self.baseline_policy = None#torch.load(None, weights_only=False)
        self.baseline_policy.to(self.device)
        self.baseline_policy.eval()
        for param in self.baseline_policy.parameters():
            param.requires_grad = False

        self.lambda_bc_init = 0.1
        self.lambda_bc_decay = 0.98

        # Normalize value for each reward
        self.beta = 0.999
        self.epsilon = 1e-5
        self.use_baseline = False

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
                    if self.use_baseline:
                        bc_dist = self.baseline_policy(rollout_data.observations[task_indices])
                        bc_loss = F.binary_cross_entropy_with_logits(logits, bc_dist, reduction="mean")
                    else:
                        bc_loss = torch.tensor(0.0, device=self.device)
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

        return True


def train_agent(env_id, hand_type, task_type, total_timesteps=TRAIN_EPSISODES, param_dict=dict(), verbose=1, provided_envs=None):
    if provided_envs is None:
        envs = make_vec_env(lambda: _make_env(env_id, hand_type, task_type), n_envs=CPU_COUNT, vec_env_cls=SubprocVecEnv)
    else:
        envs = provided_envs

    env_folder = ENV_FOLDER_2D if env_id == ENV_2D else ENV_FOLDER_2_5D
    checkpoint_folder = f"{CHECKPOINTS_FOLDER}/{env_folder}/task{task_type}/hand{hand_type}"
    tensorboard_log   = f"{LOGS_FOLDER}/{env_folder}/task{task_type}/hand{hand_type}"
    model_folder = f"{MODEL_FOLDER}/{env_folder}"
    model_path = f"{model_folder}/task{task_type}_hand{hand_type}.zip"
    os.makedirs(checkpoint_folder, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)

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
    checkpoint_callback = CheckpointCallback(save_freq=1e4, save_path=checkpoint_folder, name_prefix=MODEL_NAME, verbose=verbose)
    
    # Load or create model
    ppo_args = PPO_ARGS.copy()
    ppo_args["verbose"] = verbose
    ppo_args["tensorboard_log"] = tensorboard_log
    
    model = PPO(policy="MlpPolicy", env=envs, **ppo_args)

    # Train model
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])
    model.save(model_path)
    return model


def test_agent(env_id, hand_type, task_type, n_eval_episodes=EVAL_EPISODES, n_displayed_episodes=5, checkpoint=True):

    env_folder = ENV_FOLDER_2D if env_id == ENV_2D else ENV_FOLDER_2_5D

    # Find latest checkpoint
    if checkpoint:
        checkpoint_files = glob.glob(f"{CHECKPOINTS_FOLDER}/{env_folder}/task{task_type}/hand{hand_type}/{MODEL_NAME}_*.zip")
        if not checkpoint_files:
            raise FileNotFoundError("No saved model or checkpoints found.")
        model_path = max(checkpoint_files, key=os.path.getctime)

    # Find the specific model
    else:
        model_path = f"{MODEL_FOLDER}/{env_folder}/task{task_type}_hand{hand_type}.zip"

    env = make_vec_env(lambda: _make_env(env_id, hand_type, task_type), n_envs=1, vec_env_cls=DummyVecEnv)

    # Load model
    print(f"Loading model from {model_path}")
    model = PPO.load(env=env, path=model_path, **PPO_ARGS.copy())

    # Evaluate model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Render the environment
    gym.logger.min_level = logging.ERROR
    env = _make_env(env_id, hand_type, task_type, render=True, record=True)
    episodes = 1
    obs, info = env.reset(options={"object_type": task_type})
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, terminated, info = env.step(action)
        env.render()
        if done or terminated:
            if episodes >= n_displayed_episodes:
                break
            obs, info = env.reset(options={"object_type": task_type})
            episodes += 1
    env.close()
    print(f"Video saved to {VIDEOS_FOLDER}/{env_folder}/task{task_type}/hand{hand_type} folder.")


def param_sweep(env_id, hand_type, task_type, param_dict=None, timesteps_per_param=8e6):
    envs = make_vec_env(lambda: _make_env(env_id, hand_type, task_type), n_envs=CPU_COUNT, vec_env_cls=SubprocVecEnv)
    if param_dict is None:
        param_dict = PARAM_DICT
    param_combinations = get_param_combinations(param_dict)
    best_performance = -float("inf")
    best_params = None
    for idx, params in enumerate(param_combinations):
        print(f"{idx+1}/{len(param_combinations)} Training with {params}", end="")
        model = train_agent(env_id, hand_type, task_type, total_timesteps=timesteps_per_param, param_dict=params, verbose=0, provided_envs=envs)
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
        checkpoint_files = glob.glob(f"{CHECKPOINTS_FOLDER}/task{task_type}/hand{hand_type}/{MODEL_NAME}_*.zip")
        if not checkpoint_files:
            raise FileNotFoundError("No saved model or checkpoints found.")
        model_path = max(checkpoint_files, key=os.path.getctime)

    # Find the specific model
    else:
        model_path = f"{MODEL_FOLDER}/task{task_type}_hand{hand_type}.zip"
    
    env = make_vec_env(lambda: _make_env(env_id, hand_type, task_type), n_envs=1, vec_env_cls=DummyVecEnv)
    env_folder = ENV_FOLDER_2D if env_id == ENV_2D else ENV_FOLDER_2_5D

    # Load model
    print(f"Loading model from {model_path}")
    model = DAPG.load(env=env, path=model_path, **PPO_ARGS.copy(), tensorboard_log=f"{LOGS_FOLDER}/hand_type_{hand_type}")

    # Save the model
    baseline_path = f"{MODEL_FOLDER}/{env_folder}/task{task_type}_hand{hand_type}_{BASELINE_NAME}"
    torch.save(model, baseline_path)
    print(f"Baseline model saved to {baseline_path}")


def get_video(env_id, hand_type, task_type, ga_index, pt_index):

    env_folder = ENV_FOLDER_2D if env_id == ENV_2D else ENV_FOLDER_2_5D

    # Get videos of speicific iterations of genetic algorithm or permutation testing
    iteration_path = None
    iteration = None
    if ga_index is not None:
        iteration_path = GA_FOLDER
        iteration = ga_index
    elif pt_index is not None:
        iteration_path = f"{PT_FOLDER}/task{task_type}/hand{hand_type}"
        iteration = pt_index
    elif ga_index is not None and pt_index is not None:
        raise ValueError("Cannot specify both ga_index and pt_index.")

    # Load the model
    if iteration_path is not None:
        model_path = f"{MODEL_FOLDER}/{env_folder}/{iteration_path}/{MODEL_NAME}"
    else:
        model_path = f"{MODEL_FOLDER}/{env_folder}/task{task_type}_hand{hand_type}"

    env = make_vec_env(lambda: _make_env(env_id, hand_type, task_type, ga_index=ga_index, pt_index=pt_index, render=True, record=True), n_envs=1, vec_env_cls=DummyVecEnv)

    # Load model
    print(f"Loading model from {model_path}")
    model = PPO.load(env=env, path=model_path, **PPO_ARGS.copy())

    # If using a genetic algorithm or permutation testing, load the specified hand params
    if iteration_path is not None:
        if pt_index is not None:
            raw_params = np.load(f"{MODEL_FOLDER}/{env_folder}/{iteration_path}/{iteration}.npy")
        else:
            raw_params = np.load("morphologies/morphologies.npy")[0]
        hand_params = unnorm_hand_params(raw_params.copy())
        print(f"Norm. Params: {raw_params}")
        print("Hand Params: ", hand_params.segment_lengths, hand_params.joint_angle, hand_params.rotation_max)

    # Render the environment
    gym.logger.min_level = logging.ERROR
    env = _make_env(env_id, hand_type, task_type, ga_index=ga_index, pt_index=pt_index, render=True, record=True)
    options = {"object_type": task_type}
    if iteration_path is not None:
        options["hand_parameters"] = hand_params
    obs, info = env.reset(options=options)
    while True: # Loop until the user is satified with the video
        episodes = 1
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, terminated, info = env.step(action)
            env.render()
            if done or terminated:
                break

        # Ask the user if they want regenerate the video
        while True:
            obs, info = env.reset(options=options)
            user_input = input("Do you want to generate a new video? (y/n): ").strip().lower()
            if user_input == 'y':
                break
            else:
                return


def get_photo(env_id, hand_type, task_type, ga_index, pt_index):

    env_folder = ENV_FOLDER_2D if env_id == ENV_2D else ENV_FOLDER_2_5D

    # Get videos of speicific iterations of genetic algorithm or permutation testing
    iteration_path = None
    iteration = None
    if ga_index is not None:
        iteration_path = GA_FOLDER
        iteration = ga_index
    elif pt_index is not None:
        iteration_path = f"{PT_FOLDER}/task{task_type}/hand{hand_type}"
        iteration = pt_index
    elif ga_index is not None and pt_index is not None:
        raise ValueError("Cannot specify both ga_index and pt_index.")

    env = make_vec_env(lambda: _make_env(env_id, hand_type, -1, ga_index=ga_index, pt_index=pt_index, render=True), n_envs=1, vec_env_cls=DummyVecEnv)

    # If using a genetic algorithm or permutation testing, load the specified hand params
    if iteration_path is not None:
        if pt_index is not None:
            raw_params = np.load(f"{MODEL_FOLDER}/{env_folder}/{iteration_path}/{iteration}.npy")
        else:
            raw_params = np.load(f"{GA_MORPHOLOGIES}/morphologies.npy")[0]
        hand_params = unnorm_hand_params(raw_params.copy())
        print(f"Norm. Params: {raw_params}")
        print("Hand Params: ", hand_params.segment_lengths, hand_params.joint_angle, hand_params.rotation_max)

    # Render the environment
    gym.logger.min_level = logging.ERROR
    env = _make_env(env_id, hand_type, -1, ga_index=ga_index, pt_index=pt_index, render=True)
    options = {"object_type": -1, "photo_mode": True}
    if iteration_path is not None:
        options["hand_parameters"] = hand_params

    obs, info = env.reset(options=options)
    action = np.zeros(env.action_space.shape, dtype=np.float32)  # No action needed for photo
    action[-1] = 1.0
    frame = 0
    while True:
        obs, reward, done, terminated, info = env.step(action)
        frame_img = env.render()
        frame += 1
        if done or terminated or frame >= PHOTO_FRAME:
            photo_folder = f"{PHOTOS_FOLDER}/{env_folder}/{iteration_path}" if iteration_path is not None else f"{PHOTOS_FOLDER}/{env_folder}/task{task_type}/hand{hand_type}"
            os.makedirs(photo_folder, exist_ok=True)
            photo_path = os.path.join(photo_folder, f"photo.png")
            Image.fromarray(frame_img).save(photo_path)
            print(f"Photo saved to {photo_path}")
            obs, info = env.reset(options=options)
            break
