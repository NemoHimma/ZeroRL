import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from ..networks.dqn_model import DQN

class DQNAgent(object):
    def __init__(self,args, env, log_dir, device):

        # args
        self.device = device         # device
        self.noisy = args.noisy      # flag
        self.nsteps = args.nsteps    #  default: 1
        self.sigma_init = args.sigma_init # default: 0.5

        self.lr = args.lr                       # 1e-3
        self.gamma = args.gamma                 # 0.9
        self.batch_size = args.batch_size       # 64

        
        self.target_update_freq = args.target_update_freq
        self.memory_size = args.memory_size     # 1e6
        
        self.epsilon_start = args.epsilon_start             # 0.95
        self.epsilon_end = args.epsilon_end                 # 0.00
        self.epsilon_decay = args.epsilon_decay             # 0.01 
        
        self.env = env
        
        


        pass

    def declare_networks(self):
        pass

    def declare_memory(self):
        pass

    def get_action(self):
        pass

    def prepare_minibatch(self):
        pass

    def compute_loss(self):
        pass

    def update(self):
        pass


