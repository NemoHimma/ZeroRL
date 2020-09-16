import gym
from gym.spaces import Box
from baselines.common.atari_wrappers import *
from baselines.common.wrappers import TimeLimit

class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)

def PrepareAtariEnv(env_id, log_dir):
    env = gym.make(env_id)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = TimeLimit(env, max_episode_steps=1000)
    env = bench.Monitor(env, os.path.join(log_dir, env_id))
    env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    env = ScaledFloatFrame(env)
    env = ClipRewardEnv(env)
    env = WrapPyTorch(env)
    return env

