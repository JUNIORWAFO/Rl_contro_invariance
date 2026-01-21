from gymnasium.envs.registration import register

register(
    id="LinearUpdateEnv/GridWorld-v0",
    entry_point="LinearUpdateEnv.envs:GridWorldEnv",
)
