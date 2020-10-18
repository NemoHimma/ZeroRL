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
        self.target_noise = 0.02
        
        self.learn_start_steps = 4000 # 20000steps, 10episodes
        self.start_to_exploit_steps = 3000
        
        self.action_noise = 0.02
        self.update_freq = 5
        self.policy_decay = 5

        ### save or log
        self.save_model_freq = 50 #episodes
        self.print_freq = 5 # episodes
    
    

    def LinearDecayActionNoise(self, episode):
        action_noise = self.action_noise * (1.0 - (episode/ float(self.num_episodes)))
        return action_noise

    def LinearDecayTargetNoise(self, episode):
        target_noise = self.target_noise * (1.0 - (episode/float(self.num_episodes)))
        return target_noise

    



