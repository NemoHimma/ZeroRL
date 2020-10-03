import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBody(nn.Module):
    pass


class MLPBody(nn.Module):
    '''input_shape: (obs_dims, ) Assuming Mujoco States Input
       forward_output: (critic_value, actor_features) Distangled Representation of Actor-Critic Structure
    '''

    def __init__(self, input_shape, hidden_size = 64):
        
        init_ = lambda m: self.init(m , nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        # Distangled representation for actor-critic
        self.actor_layer1 = init_(nn.Linear(input_shape, hidden_size))
        self.actor_layer2 = init_(nn.Linear(hidden_size, hidden_size))

        self.critic_layer1 = init_(nn.Linear(input_shape, hidden_size))
        self.critic_layer2 = init_(nn.Linear(hidden_size, hidden_size))
        self.critic_layer3 = init_(nn.Linear(hidden_size, 1))

        # Shared Representation for actor-critic
        #self.ac_layer1 = init_(nn.Linear(input_shape, hidden_size))
        #self.ac_layer2 = init_(nn.Linear(hidden_size, hidden_size))
        #self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()
    
    def forward(self, x):
        actor_features = F.tanh(self.actor_layer1(x))
        actor_features = F.tanh(self.actor_layer2(actor_features))

        critic_value = F.tanh(self.critic_layer1(x))
        critic_value = F.tanh(self.critic_layer2(critic_value))
        critic_value = self.critic_layer3(critic_value)

        return critic_value, actor_features

    def init(module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module

    @property
    def actor_features_size(self):
        return hidden_size
        
