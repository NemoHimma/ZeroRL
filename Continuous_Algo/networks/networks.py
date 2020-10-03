import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.distributions import DiagGassian
from .bodies import MLPBody, CNNBody


class ActorCritic(nn.Module):

    def __init__(self, obs_shape, action_space, body = MLPBody):
        super(ActorCritic, self).__init__()
        
        self.input_shape = obs_shape[0]
        self.num_actions = action_space.shape[0]
        
        self.body = body(self.input_shape)  # MLPBody return critic_value & actor_features
        self.dist = DiagGassian(self.body.actor_features_size, self.num_actions)  # Map actor_features to output_distribution

    def forward(self, x):
        critic_value, actor_features = self.body(x)
        dist = self.dist(actor_features)

        return critic_value, dist


    
        






