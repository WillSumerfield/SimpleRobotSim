"""
Play the game manually using the keyboard
"""

import gymnasium as gym
import numpy as np
from pynput import keyboard

import Grasper
from Grasper.wrappers import BetterExploration
from key_checks import key_presses, get_actions, on_press, on_release
from hand_morphologies import HAND_TYPES


def manual_control(env, hand_type):
    global key_presses

    options = {"hand_parameters": HAND_TYPES[hand_type]}

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