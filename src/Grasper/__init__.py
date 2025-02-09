from gymnasium.envs.registration import register

register(
    id="Grasper/Grasper-v0",
    entry_point="Grasper.envs:GrasperEnv",
)
