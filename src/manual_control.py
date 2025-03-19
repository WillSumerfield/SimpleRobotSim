"""
Play the game manually using the keyboard
"""

import gymnasium as gym
import numpy as np
from pynput import keyboard

import Grasper


key_presses = {
    "w": False,
    "a": False,
    "s": False,
    "d": False,
    "space": False,
    "esc": False,
    "enter": False,
}


def manual_control(env):
    global key_presses

    # Choose env
    if env == "manipulation":
        env_name = "Grasper/Manipulation-v0"
    elif env == "claw_game":
        env_name = "Grasper/ClawGame-v0"
    else:
        raise ValueError("Invalid environment")

    env = gym.make(env_name, render_mode="human")
    env.reset()

    # Start a separate pynput listener thread
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Repeat until the env is complete
    while True:
        # Quit the game
        if key_presses["esc"]:
            break

        # Handle player input
        movement = 0 # Default to 'up'
        if key_presses["s"]:
            movement = 1
        if key_presses["a"]:
            movement = 2
        if key_presses["d"]:
            movement = 3

        claw = 0 # Default to opening the claw
        if key_presses["space"]:
            claw = 1
        
        # Reset the game
        if key_presses["enter"]:
            env.reset()

        # Take a step in the environment
        obs, reward, done, _, info = env.step(np.array([movement, claw]))
        env.render()
        if done:
            env.reset()

    env.close()


def on_press(key):
        global key_presses

        try:
            # Player inputs
            if key.char == "w":
                key_presses["w"] = True
            if key.char == "s":
                key_presses["s"] = True
            if key.char == "a":
                key_presses["a"] = True
            if key.char == "d":
                key_presses["d"] = True
        
        except AttributeError:
            if key == keyboard.Key.space:
                key_presses["space"] = True
            if key == keyboard.Key.esc:
                key_presses["esc"] = True
            if key == keyboard.Key.enter:
                key_presses["enter"] = True


def on_release(key):
        global key_presses

        try:
            # Player inputs
            if key.char == "w":
                key_presses["w"] = False
            if key.char == "s":
                key_presses["s"] = False
            if key.char == "a":
                key_presses["a"] = False
            if key.char == "d":
                key_presses["d"] = False
        
        except AttributeError:
            if key == keyboard.Key.space:
                key_presses["space"] = False
            if key == keyboard.Key.esc:
                key_presses["esc"] = False
            if key == keyboard.Key.enter:
                key_presses["enter"] = False