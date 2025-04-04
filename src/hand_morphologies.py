import numpy as np

from Grasper.envs.manipulation import Hand


HAND_TYPES = [
    # Claw
    Hand.Parameters(
        np.array([48]*4), # Segment lengths
        np.array([np.pi/4]*2), # Joint angles
        np.pi * (15/16.0) # Rotation Max
    ),
    # Pincer
    Hand.Parameters(
        np.array([36]*4), # Segment lengths
        np.array([0]*2), # Joint angles
        np.pi * (35/32.0) # Rotation Max
    ),
    # Short Elbows
    Hand.Parameters(
        np.array([64, 64, 48, 48]), # Segment lengths
        np.array([np.pi * (3/4.0)]*2), # Joint angles
        np.pi * (3/4.0) # Rotation Max
    ),
    # Long Elbows
    Hand.Parameters(
        np.array([64, 64, 32, 32]), # Segment lengths
        np.array([np.pi * (3/4.0)]*2), # Joint angles
        np.pi * (3/4.0) # Rotation Max
    ),
    # Basket and Sweeper
    Hand.Parameters(
        np.array([48, 96, 48, 96]), # Segment length
        np.array([np.pi/2, np.pi*(3/4.0)]), # Joint angles
        np.pi * (3/4.0) # Rotation Max
    ),
    # Stubby
    Hand.Parameters(
        np.array([16, 16, 64, 64]), # Segment length
        np.array([np.pi*(13/32.0)]*2), # Joint angles
        np.pi * (3/4.0) # Rotation Max
    ),
]