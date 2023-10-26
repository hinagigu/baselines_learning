from gymnasium.envs.registration import register

register(

    id="envs/obs-v0",
    entry_point="envs.obstacle_gridworld:Obsworld"
)

register(
    id="envs/obs-v1",
    entry_point="envs.obs_world:obs_world"
)
