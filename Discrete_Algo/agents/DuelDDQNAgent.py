import torch
import pickle
import torch.optim as optim
import numpy as np
import csv
import os 

from agents.DQNAgent import DQNAgent
from buffer.ReplayBuffer import ExperienceReplayBuffer
from networks.networks import DuelingDQN
from networks.bodies import AtariBody

class DuelDDQNAgent(DQNAgent):
    def __init__(self, config, env, log_dir, static_policy=False):
        super(DuelDDQNAgent, self).__init__(config=config, env=env, log_dir=log_dir, static_policy=static_policy)

    def declare_networks(self):
        self.model = DuelingDQN(self.input_shape, self.num_actions, self.noisy, self.sigma_init, body = AtariBody)
        self.target_model = DuelingDQN(self.input_shape, self.num_actions, self.noisy, self.sigma_init, body = AtariBody)

    # Seperate Action Selection & Evaluation   Double 
    def get_max_next_state_action(self, non_final_next_states):
        max_next_action = self.model(non_final_next_states).max(dim=1)[1].view(-1, 1)
        return max_next_action

        



    