import os
import numpy as np 

import gym
from gym import spaces
from gym.spaces.box import Box

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
        self.observation_space.low[0, 0, 0], self.observation_space.high[0, 0, 0], [obs_shape[2],obs_shape[1], obs_shape[0]],dtype = self.observation_space.dtype)
         
    def observation(self, observation):
        return observation.transpose(2, 0, 1)
