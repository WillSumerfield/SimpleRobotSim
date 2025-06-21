"""
Play the game manually using the keyboard
"""

import gymnasium as gym
import numpy as np
from pynput import keyboard

from src import TASK_2D, TASK_2_5D
import src.Grasper
from src.Grasper.wrappers import BetterExploration
from src.key_checks import key_presses, get_actions, on_press, on_release
from src.hand_morphologies import HAND_TYPES, unnorm_hand_params


GA_MORPHOLOGIES = "morphologies/morphologies.npy"


def manual_control(env_name, hand_type, task_type, ga_index):
    global key_presses

    if ga_index is not None:
        all_morphs = np.load(GA_MORPHOLOGIES)
        hand_parameters = unnorm_hand_params(all_morphs[ga_index].copy())
    else:
        hand_parameters = HAND_TYPES[hand_type]
    options = {"hand_parameters": hand_parameters, "object_type": task_type}

    env = gym.make(env_name, render_mode="human")
    if env == TASK_2D:
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
        
        actions = None
        if env_name == TASK_2D:
            # Only take one movement action at a time
            move = 0
            for i, m in enumerate(movement):
                if m:
                    move = i+1
                    break
            actions = np.array([move, rotation, open_hand])
        elif env_name == TASK_2_5D:
            moves = np.zeros(2, dtype=np.float32)
            if movement[1]:
                moves[0] = -1.0
            if movement[3]:
                moves[0] = 1.0
            if movement[0]:
                moves[1] = 1.0
            if movement[2]:
                moves[1] = -1.0

            if open_hand == 2:
                open_hand = -1.0
            actions = np.concatenate([moves, np.array([open_hand, open_hand])])

        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(actions)
        env.render()
        if terminated or truncated:
            env.reset(options=options)

    env.close()