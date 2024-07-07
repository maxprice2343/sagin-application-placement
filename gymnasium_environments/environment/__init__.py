from gymnasium.envs.registration import register

register(
    id = "environments/ApplicationPlacementEnv-v0",
    entry_point = "environments.envs:ApplicationPlacementEnv",
    max_episode_steps=500
)