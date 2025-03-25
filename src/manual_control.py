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
    "q": False,
    "e": False,
    "space": False,
    "esc": False,
    "enter": False,
}


def manual_control(env):
    global key_presses

    env = gym.make(env, render_mode="human")
    env = Grasper.wrappers.BetterExploration(env)
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
        if key_presses["w"]:
            movement = 1
        if key_presses["s"]:
            movement = 2
        if key_presses["a"]:
            movement = 3
        if key_presses["d"]:
            movement = 4

        # Rotation
        rotation = 0
        if key_presses["q"]:
            rotation = 1
        if key_presses["e"]:
            rotation = 2

        open_hand = 0 # Default to opening the claw
        if key_presses["space"]:
            open_hand = 1
        
        # Reset the game
        if key_presses["enter"]:
            env.reset()

        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(np.array([movement, rotation, open_hand]))
        env.render()
        if terminated or truncated:
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
            if key.char == "e":
                key_presses["e"] = True
            if key.char == "q":
                key_presses["q"] = True
        
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
            if key.char == "e":
                key_presses["e"] = False
            if key.char == "q":
                key_presses["q"] = False
        
        except AttributeError:
            if key == keyboard.Key.space:
                key_presses["space"] = False
            if key == keyboard.Key.esc:
                key_presses["esc"] = False
            if key == keyboard.Key.enter:
                key_presses["enter"] = False