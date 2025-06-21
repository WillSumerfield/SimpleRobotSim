from gymnasium.envs.registration import register

register(
    id="Grasper/Grasp2D-v0",
    entry_point="src.Grasper.envs:Grasp2DEnv",
)

register(
    id="Grasper/Grasp2.5D-v0",
    entry_point="src.Grasper.envs:Grasp2_5DEnv",
)