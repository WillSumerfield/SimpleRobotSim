from pynput import keyboard


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


# Get actions from keyboard inputs
def get_actions():
    global key_presses

    # Movement
    movement = [key_presses["w"], key_presses["a"], key_presses["s"], key_presses["d"]]

    # Rotation
    rotation = 0
    if key_presses["q"]:
        rotation = 1
    if key_presses["e"]:
        rotation = 2

    # Open/close hand
    open_hand = 2 # Default to opening the claw
    if key_presses["space"]:
        open_hand = 1

    return movement, rotation, open_hand


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