import random 
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ExperienceReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size), None , None
    
    def __len__(self):
        return len(self.memory)
