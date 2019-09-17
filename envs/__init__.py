from gym.envs.registration import register

register(
    id='Draw2D-v0',
    entry_point='envs.draw2d_env:Draw2DEnv',
    timestep_limit = 100
)