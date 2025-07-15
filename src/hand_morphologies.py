import numpy as np

from src import *


HAND_TYPES = [
    # Claw
    np.concatenate((
        np.array([48]*4), # Segment lengths
        np.array([np.pi/4]*2), # Joint angles
        np.array([np.pi * (15/16.0)]) # Rotation Max
    )),
    # Pincer
    np.concatenate((
        np.array([36]*4), # Segment lengths
        np.array([0]*2), # Joint angles
        np.array([np.pi * (35/32.0)]) # Rotation Max
    )),
    # Short Elbows
    np.concatenate((
        np.array([64, 64, 64, 64]), # Segment lengths
        np.array([np.pi * (3/4.0)]*2), # Joint angles
        np.array([np.pi * (3/4.0)]) # Rotation Max
    )),
    # Long Elbows
    np.concatenate((
        np.array([96, 96, 80, 80]), # Segment lengths
        np.array([np.pi * (3/4.0)]*2), # Joint angles
        np.array([np.pi * (3/4.0)]) # Rotation Max
    )),
    # Basket and Sweeper
    np.concatenate((
        np.array([48, 96, 48, 96]), # Segment length
        np.array([np.pi/2, np.pi*(3/4.0)]), # Joint angles
        np.array([np.pi * (3/4.0)]) # Rotation Max
    )),
    # Stubby
    np.concatenate((
        np.array([16, 16, 64, 64]), # Segment length
        np.array([np.pi*(13/32.0)]*2), # Joint angles
        np.array([np.pi * (3/4.0)]) # Rotation Max
    )),
]

HAND_PARAM_MINS = np.array([1,             # Segment lengths
                            -np.pi,        # Joint angles
                            np.pi * (1/2)  # Rotation Max
                  ])
HAND_PARAM_MAXS = np.array([128,             # Segment lengths
                            np.pi,           # Joint angles
                            np.pi * (3/2.0)  # Rotation Max
                  ])
HAND_PARAM_MEANS = (HAND_PARAM_MINS + HAND_PARAM_MAXS) / 2.0
HAND_PARAM_STDS = (HAND_PARAM_MAXS - HAND_PARAM_MINS) / 6.0  # Limits are 3 stds away from the mean


def norm_hand_params(hand_parameters: np.ndarray) -> np.ndarray:
    hand_parameters[0:4] = (hand_parameters[0:4] - HAND_PARAM_MEANS[0]) / HAND_PARAM_STDS[0]  # Segment lengths
    hand_parameters[4:6] = (hand_parameters[4:6] - HAND_PARAM_MEANS[1]) / HAND_PARAM_STDS[1]  # Joint angles
    hand_parameters[6]   = (hand_parameters[6]   - HAND_PARAM_MEANS[2]) / HAND_PARAM_STDS[2]  # Rotation Max
    return hand_parameters


def unnorm_hand_params(hand_parameters: np.ndarray) -> np.ndarray:
    hand_parameters = hand_parameters.copy()
    hand_parameters[0:4] = np.clip(hand_parameters[0:4]*HAND_PARAM_STDS[0] + HAND_PARAM_MEANS[0], HAND_PARAM_MINS[0], HAND_PARAM_MAXS[0]) # Segment lengths
    hand_parameters[4:6] = np.clip(hand_parameters[4:6]*HAND_PARAM_STDS[1] + HAND_PARAM_MEANS[1], HAND_PARAM_MINS[1], HAND_PARAM_MAXS[1]) # Joint angles
    hand_parameters[6]   = np.clip(hand_parameters[6]*HAND_PARAM_STDS[2]   + HAND_PARAM_MEANS[2], HAND_PARAM_MINS[2], HAND_PARAM_MAXS[2]) # Rotation Max
    return hand_parameters
