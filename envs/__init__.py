from gymnasium.envs.registration import register

register(
    id="envs/obs-v0",
    entry_point="obstacle_gridworld:obsworld"
)
