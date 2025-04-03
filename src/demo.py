"""
Provide a demo for the Grasper environments manually using the keyboard.
"""

import gymnasium as gym
import numpy as np
from pynput import keyboard
import time
import os
import glob

import Grasper
from Grasper.wrappers import BetterExploration
from key_checks import key_presses, get_actions, on_press, on_release


DEMO_FOLDER = "./demos"


def provide_demos(env_name):
    env = gym.make(env_name, render_mode="human")
    env = BetterExploration(env)
    env.reset()

    # Start a separate pynput listener thread
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    if not os.path.exists(DEMO_FOLDER):
        os.makedirs(DEMO_FOLDER)
    if not os.path.exists(DEMO_FOLDER + "/" + env.spec.id):
        os.makedirs(DEMO_FOLDER + "/" + env.spec.id)

    # Find how many demos are in the demo folder
    demo_files = glob.glob(DEMO_FOLDER + "/" + env.spec.id + "/demo_*.npy")
    demo_file_count = len(demo_files)

    # Make a new file in the demo folder
    demo_file = DEMO_FOLDER + "/" + env.spec.id + f"/demo_{demo_file_count+1}.npy"
    demo_seeds_file = DEMO_FOLDER + "/" + env.spec.id + f"/demo_seeds_{demo_file_count+1}.npy"

    # Repeat until the user is done recording demos
    demo_count = 1
    obs_space = env.unwrapped.observation_space.shape[0]
    action_space = env.unwrapped.action_space.shape[0]
    demo_matrix = np.empty((0, obs_space + action_space, env.unwrapped.MAX_TIME), dtype=np.float32)
    demo_seeds = np.empty((0), dtype=np.int32)
    while True:
        print("\nPress 'Space' to begin recording a demo. Press 'Esc' to quit.\n")

        # Wait for the user to press 'Space' or 'Esc'
        while not key_presses["space"] and not key_presses["esc"]:
            pass
        if key_presses["esc"]:
            break
        time.sleep(0.5)
        print(f"Recording demo {demo_count}...", end="\r")
        seed, trajectory, timesteps = record_demo(env)
        print(f"Demo {demo_count} recorded in {timesteps} steps.")

        # Ensure the demo is quality
        time.sleep(0.5)
        print(f"Keep demo {demo_count} ('Space') or not ('Esc')?")

        # Wait for the user to press 'Space' or 'Esc'
        while not key_presses["space"] and not key_presses["esc"]:
            pass
        if key_presses["esc"]:
            print(f"Demo discarded.")
        else:
            demo_matrix = np.append(demo_matrix, trajectory, axis=0)
            demo_seeds = np.append(demo_seeds, seed)
            np.save(demo_file, demo_matrix)
            np.save(demo_seeds_file, demo_seeds)
            print(f"Demo saved.")
            demo_count += 1
        print(f"Total Demos Recorded: {demo_count-1}")
        time.sleep(0.5)

    env.close()


def record_demo(env):
    # Run until the env. is completed
    seed = np.random.randint(0, 2**32 - 1)
    obs, info = env.reset(seed=seed)
    step = 0
    obs_space = env.unwrapped.observation_space.shape[0]
    action_space = env.unwrapped.action_space.shape[0]
    trajectory = np.zeros((1, obs_space + action_space, env.unwrapped.MAX_TIME), dtype=np.float32)
    while True:
        movement, rotation, open_hand = get_actions()
        action = np.array([movement, rotation, open_hand])
        trajectory[0, :obs_space, step] = obs
        trajectory[0, obs_space:, step] = action
        step += 1
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            return seed, trajectory, step+1
        

def load_demos(env_id, demo_number):
    demo_file = glob.glob(DEMO_FOLDER + "/" + env_id + f"/demo_{demo_number}.npy")
    if not demo_file:
        print("No demo collection found of that number.")
        raise FileNotFoundError("No demo collection found of that number.")
    demo_matrix = np.load(demo_file[0])
    demo_seeds_file = demo_file[0].replace("demo_", "demo_seeds_")
    demo_seeds = np.load(demo_seeds_file)
    print(f"Loaded {len(demo_seeds)} demos from {demo_file[0]}")
    return demo_seeds, demo_matrix


def play_demos(env_name, demo_number):
    env = gym.make(env_name, render_mode="human")
    env = BetterExploration(env)

    demo_seeds, demo_matrix = load_demos(env_name, demo_number)

    # Play the demos
    for seed, demo in zip(demo_seeds, demo_matrix):
        play_demo(env, seed, demo)
    env.close()


def play_demo(env, seed, demo):
    obs, info = env.reset(seed=int(seed))
    obs_space = env.unwrapped.observation_space.shape[0]
    for step in range(demo.shape[1]):
        obs = demo[:obs_space, step]
        action = demo[obs_space:, step]
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            break