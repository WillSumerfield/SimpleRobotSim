"""
Play the game manually using the keyboard
"""

import gymnasium as gym
import numpy as np
from pynput import keyboard

import Grasper
from Grasper.wrappers import BetterExploration
from key_checks import key_presses, get_actions, on_press, on_release
from hand_morphologies import HAND_TYPES, unnorm_hand_params


def manual_control(env, hand_type, task_type, ga_index):
    global key_presses

    if ga_index is not None:
        raw_params = np.load(f"checkpoints/genetic_algorithm/{ga_index}/hand_params.npy")
        hand_parameters = unnorm_hand_params(raw_params.copy())
    else:
        hand_parameters = HAND_TYPES[hand_type]
    options = {"hand_parameters": hand_parameters, "object_type": task_type}

    env = gym.make(env, render_mode="human")
    env = BetterExploration(env)
    env.reset(options=options)

    # Start a separate pynput listener thread
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Repeat until the env is complete
    while True:
        # Quit the game
        if key_presses["esc"]:
            break

        movement, rotation, open_hand = get_actions()
        
        # Reset the game
        if key_presses["enter"]:
            env.reset(options=options)

        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(np.array([movement, rotation, open_hand]))
        env.render()
        if terminated or truncated:
            env.reset(options=options)

    env.close()