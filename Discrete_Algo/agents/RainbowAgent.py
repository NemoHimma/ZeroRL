import torch
import pickle
import torch.optim as optim
import numpy as np
import csv
import os 

from agents.C51DuelAgent import C51DuelAgent
from buffer.ReplayBuffer import PrioritizedReplayMemory
from networks.networks import CategoricalDuelingDQN
from networks.bodies import AtariBody

# Categorical, Dueling, N-Steps (C51DuelAgent) + Double + Noisy + PER

# Double modify the get_max_next_state_action
# Noisy means set config.USE_NOISY_NETS & config.USE_PRIORITY_REPLAY
# PER import the New Buffer


class RainbowAgent(C51DuelAgent):
    def __init__(self, config, env, log_dir, static_policy=False):
        super(RainbowAgent, self).__init__(config, env, log_dir, static_policy=False)

    def declare_memory(self):
        self.memory = PrioritizedReplayMemory(self.replay_buffer_size, self.alpha, self.priority_beta_start, self.priority_beta_frames)

    def get_action(self, s, eps):
        with torch.no_grad():
                X = torch.tensor([s], device=self.device, dtype=torch.float) 
                self.model.sample_noise()
                a = self.model(X) * self.supports
                a = a.sum(dim=2).max(1)[1].view(1, 1)
                return a.item()

    def get_max_next_state_action(self, next_states):
        next_dist = self.model(next_states) * self.supports # Double Part
        return next_dist.sum(dim=2).max(1)[1].view(next_states.size(0), 1, 1).expand(-1, -1, self.atoms)



