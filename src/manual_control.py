"""
Play the game manually using the keyboard
"""

import time

import gymnasium as gym
import numpy as np
from pynput import keyboard

import Grasper


FPS = 1/60.0

key_presses = {
    "w": False,
    "a": False,
    "s": False,
    "d": False,
    "space": False,
    "esc": False
}


def manual_control():
    global key_presses

    env = gym.make('Grasper/Grasper-v0', render_mode="human")
    env.reset()

    # Start a separate pynput listener thread
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Repeat until the env is complete
    while True:
        time.sleep(FPS)

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