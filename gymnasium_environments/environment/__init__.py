from gymnasium.envs.registration import register

register(
    id = "ApplicationPlacementEnv-v0",
    entry_point = "environment.envs:ApplicationPlacementEnv",
    max_episode_steps=500
)