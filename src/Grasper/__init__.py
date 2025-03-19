from gymnasium.envs.registration import register

register(
    id="Grasper/ClawGame-v0",
    entry_point="Grasper.envs:ClawGameEnv",
)

register(
    id="Grasper/Manipulation-v0",
    entry_point="Grasper.envs:ManipulationEnv",
)