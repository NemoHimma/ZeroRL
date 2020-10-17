import torch
import math

class Config(object):
    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 1
        
        self.per_epoch_steps = 20
        self.epochs = 1000

        self.buffer_size = int(1e6) 
        self.batch_size = 64
        

        self.policy_lr = 1e-3 # actor_lr
        self.value_lr = 1e-3 # critic_lr
        self.gamma = 0.99 # discount factor
        self.polyak = 0.995 # polyak update rate
        
        self.learn_start = 1000
        self.start_to_exploit_steps = 10000
        self.update_freq = 5
        

        self.max_episode_len = 3000
        self.action_noise = 0.1

        ### save or log
        self.save_model_freq = 100
    
    

    def LinearDecayLR(self, optimizer, epoch, num_epochs, initial_lr):
        lr = initial_lr * (1.0 - (epoch/ float(num_epochs)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    



