import torch
import math

class Config(object):
    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 1
        
        self.max_episode_steps = 1000
        self.num_episodes = 2000

        self.buffer_size = int(1e5) 
        self.batch_size = 128

        self.policy_lr = 1e-4 # actor_l
        self.value_lr = 1e-4 # critic_lr
        self.gamma = 0.99 # discount factor
        self.polyak = 0.995 # polyak update rate
        
        self.learn_start_steps = int(2e4) # 20000steps, 10episodes
        self.start_to_exploit_steps = int(1e4) # 10000steps, 5episodes
        
        self.action_noise = 0.01
        self.update_freq = 5

        ### save or log
        self.save_model_freq = 50 #episodes
        self.print_freq = 5 # episodes
    
    

    def LinearDecayLR(self, optimizer, epoch, num_epochs, initial_lr):
        lr = initial_lr * (1.0 - (epoch/ float(num_epochs)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    



