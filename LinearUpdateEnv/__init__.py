from gymnasium.envs.registration import register

register(
    id="LinearUpdateEnv/LinearUpdateEnv-v0",
    entry_point="LinearUpdateEnv.envs:LinearUpdateEnv",
)
