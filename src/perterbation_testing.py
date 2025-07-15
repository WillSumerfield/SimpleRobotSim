import sys
import os

import pandas as pd
import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from src import *
from src.Grasper.wrappers import TaskType
from src.hand_morphologies import HAND_TYPES, norm_hand_params, unnorm_hand_params
from src.agent import CPU_COUNT, PPO_ARGS, DAPG
np.set_printoptions(precision=2, suppress=True)


EVAL_EPISODES = 1000
TRAIN_TIMESTEPS = 4e6
FINE_TUNE_TIMESTEPS = 1e6

PARAM_INDEX = [0, 1]
PARAM_OFFSET = 0
PARAM_SPAN = 0.5
PERMUATION_COUNT = 4
PERMUATION_SIZE = (PARAM_SPAN)/(PERMUATION_COUNT)


def _make_env(env_id, task_type):
    env = gym.make(env_id)
    env = TaskType(env, task_type)
    env = gym.wrappers.FlattenObservation(env)
    env = Monitor(env)
    check_env(env)
    return env


def perterbation_testing(env_id, hand_type, task_type, num_params=1):

    env_folder = ENV_FOLDER_2D if env_id == ENV_2D else ENV_FOLDER_2_5D

    # Redirect print statements to a log file
    results_path = f"{RESULTS_FOLDER}/{env_folder}"
    os.makedirs(results_path, exist_ok=True)
    log_file = open(f"{results_path}/{PT_RESULTS_FILE}", "w")
    log_data = {"Iteration": [], "Performance": [], "Parameters": []}

    log_file.write(f"Starting perterbation testing with hand type: {hand_type}, task type: {task_type}, num params: {num_params}")
    envs = make_vec_env(lambda: _make_env(env_id, task_type), n_envs=CPU_COUNT, vec_env_cls=SubprocVecEnv)
    
    # Get the performance of the original hand
    original_params = HAND_TYPES[hand_type]
    model_path = f"{MODEL_FOLDER}/{env_folder}/task{task_type}_hand{hand_type}.zip"
    ppo_args = PPO_ARGS.copy()
    original_model = DAPG.load(env=envs, path=model_path, **ppo_args)
    options = {"hand_parameters": original_params}
    envs.set_options(options)
    envs.reset()
    original_mean_reward, original_std_reward = evaluate_policy(original_model, envs, n_eval_episodes=EVAL_EPISODES, deterministic=True)
    log_file.write(f"Original hand parameters: {norm_hand_params(original_params)}")
    log_file.write(f"Original hand performance: {original_mean_reward:.2f} +/- {original_std_reward:.2f}")
    log_data["Iteration"] += [0],
    log_data["Performance"] += [original_mean_reward]
    log_data["Parameters"] += [norm_hand_params(original_params).tolist()[PARAM_INDEX]]
    del original_model

    # Randomly permute the hand parameters
    permuted_params = norm_hand_params(original_params)
    permuted_params[PARAM_INDEX] += PARAM_OFFSET
    permuted_hand_params = unnorm_hand_params(permuted_params)
    options = {"hand_parameters": permuted_hand_params}
    envs.set_options(options)
    envs.reset()

    # Train a controller for the permuted hand
    ppo_args = PPO_ARGS.copy()
    ppo_args["verbose"] = 0
    ppo_args["tensorboard_log"] = f"{LOGS_FOLDER}/{env_folder}/task_{task_type}/hand_type{hand_type}"
    permuted_model = DAPG(env=envs, **ppo_args)
    permuted_model.learn(total_timesteps=TRAIN_TIMESTEPS, reset_num_timesteps=True)
    permuted_model.save(f"{MODEL_FOLDER}/{env_folder}/{PT_FOLDER}/task{task_type}/hand_type{hand_type}/{MODEL_NAME}_0.zip")
    np.save(f"{MODEL_FOLDER}/{env_folder}/{PT_FOLDER}/hand_type{hand_type}task{task_type}/{HAND_PARAMS_FILE}_0.npy", permuted_params)

    # Test the permuted hand's performance
    mean_reward, std_reward = evaluate_policy(permuted_model, envs, n_eval_episodes=EVAL_EPISODES, deterministic=True)
    log_file.write(f"Permuted hand parameters: {permuted_params}")
    log_file.write(f"Permuted hand performance: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Repeatedly permute the hand parameters and pick the best one
    max_reward = mean_reward
    best_params = permuted_params.copy()
    best_model = permuted_model
    for permutation_idx in range(1, PERMUATION_COUNT+1):

        # Permute the hand parameter again, try both + and -
        for permutation_dir in range(2):
            params = permuted_params.copy()
            params[PARAM_INDEX] += (1 if permutation_dir == 0 else -1) * PERMUATION_SIZE
            permuted_hand_params = unnorm_hand_params(params)
            options = {"hand_parameters": permuted_hand_params}
            envs.set_options(options)
            envs.reset()

            # Train a controller for the newly permuted hand
            previous_model_path = f"{MODEL_FOLDER}/{env_folder}/{PT_FOLDER}/task{task_type}/hand_type{hand_type}/{MODEL_NAME}_{permutation_idx-1}.zip"
            permuted_model = DAPG.load(env=envs, path=previous_model_path, **ppo_args)
            permuted_model.learn(total_timesteps=FINE_TUNE_TIMESTEPS, reset_num_timesteps=True)

            # Test the permuted hand's performances
            mean_reward, std_reward = evaluate_policy(permuted_model, envs, n_eval_episodes=EVAL_EPISODES, deterministic=True)
            if mean_reward > max_reward:
                max_reward = mean_reward
                best_params = params
                del best_model
                best_model = permuted_model
            else:
                del permuted_model

        # Use the best hand parameters for the next iteration
        permuted_params = best_params
        best_model.save(f"{MODEL_FOLDER}/{env_folder}/{PT_FOLDER}/task{task_type}/hand_type{hand_type}/{MODEL_NAME}_{permutation_idx}.zip")
        np.save(f"{MODEL_FOLDER}/{env_folder}/{PT_FOLDER}/task{task_type}/hand_type{hand_type}/{MODEL_NAME}_{permutation_idx}.npy", permuted_params)
        log_file.write(f"Iteration: {permutation_idx}, Performance: {max_reward:.2f}, Parameters {PARAM_INDEX} = {permuted_params[PARAM_INDEX]}")
        log_data["Iteration"] += [permutation_idx],
        log_data["Performance"] += [max_reward]
        log_data["Parameters"] += [norm_hand_params(best_params).tolist()[PARAM_INDEX]]

    # Compare the performance of the original hand and the permuted hand
    log_file.write(f"Original Performance: {original_mean_reward}, Interatively Permuted Performance: {max_reward}")

    # Compare the parameters of the original hand and the permuted hand
    log_file.write(f"Original parameters: {PARAM_INDEX} = {norm_hand_params(original_params)[PARAM_INDEX]}, Interatively Permuted parameters: {PARAM_INDEX} = {permuted_params[PARAM_INDEX]}")
    df = pd.DataFrame(log_data, columns=["Iteration", "Performance"] + [f"Param_{i}" for i in PARAM_INDEX])
    df.to_csv(f"{results_path}/{PT_CSV_FILE}", index=False)

    print(f"Results saved to {results_path}/{PT_RESULTS_FILE} and {results_path}/{PT_CSV_FILE}")
    log_file.close()
