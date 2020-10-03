import torch
import math

class Config(object):
    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 1
        
        self.num_processes = 16 # aka num_agents
        self.num_steps = 5 # aka rollouts

        self.value_loss_coef = 0.5  # a2c loss
        self.entropy_loss_coef = 0.001 # a2c loss
        self.max_grad_norm  = 0.5 # a2c clipping


        self.USE_GAE = True
        self.gae_lambda = 0.95

        self.lr = 7e-4
        self.gamma = 0.99 # discount factor
        
        self.eps = 1e-5 # RMSprop epsilon
        self.alpha = 0.99 # RMSprop alpha


        self.num_envs_steps = 10e6
        self.num_updates = int(self.num_envs_steps // self.num_steps // self.num_processes)

        self.USE_DECAY_LR = False
        self.USE_PROPER_TIME_LIMITS = True
    
    def LinearDecayLR(self, optimizer, epoch, num_epochs, initial_lr):
        lr = initial_lr * (1.0 - (epoch/ float(num_epochs)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    



