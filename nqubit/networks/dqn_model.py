import torch
import torch.nn as nn
import torch.nn.functional as F

from .noisy_layer import NoisyLinear

def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1.414)
        nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    '''

    input_shape: 6
    num_actions: index(0,13)

    '''
    def __init__(self, input_shape, num_actions=13, noisy=False, sigma_init=0.5):

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.noisy = noisy
        

        if not self.noisy:
            self.fc1 = nn.Linear(self.input_size, 32)
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, self.num_actions)
        else:
            self.fc1 = NoisyLinear(self.input_size, 32, sigma_init)
            self.fc2 = NoisyLinear(32, 32, sigma_init)
            self.fc3 = NoisyLinear(32, self.num_actions, sigma_init)

        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x  #[batch_size, num_actions]


    def sample_noise(self):
        if self.noisy:
            self.fc1.sample_noise()
            self.fc2.sample_noise()
            self.fc3.sample_noise()


