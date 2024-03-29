
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


LOG_STD_MIN = -20
LOG_STD_MAX = 2
epsilon = 1e-7

# init parameters
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain = 1)
        torch.nn.init.constant_(m.bias, 0)


class TwinQNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()

        self.Q1_layer1 = nn.Linear(obs_dim + act_dim, hidden_size)
        self.Q1_layer2 = nn.Linear(hidden_size, hidden_size)
        self.Q1_value = nn.Linear(hidden_size, 1)

        self.Q2_layer1 = nn.Linear(obs_dim + act_dim, hidden_size)
        self.Q2_layer2 = nn.Linear(hidden_size, hidden_size)
        self.Q2_value = nn.Linear(hidden_size, 1)

        self.apply(weights_init_)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim = -1)

        Q1 = F.relu(self.Q1_layer1(x))
        Q1 = F.relu(self.Q1_layer2(Q1))
        Q1 = self.Q1_value(Q1)

        Q2 = F.relu(self.Q2_layer1(x))
        Q2 = F.relu(self.Q2_layer2(Q2))
        Q2 = self.Q2_value(Q2)

        return Q1, Q2



class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size, act_limit):
        super().__init__()
        self.act_limit = act_limit

        self.layer1 = nn.Linear(obs_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)

        self.mean_layer = nn.Linear(hidden_size, act_dim)
        self.log_std_layer = nn.Linear(hidden_size, act_dim)

        self.apply(weights_init_)

    def forward(self, obs, deterministic = False, with_logprob = True):
        obs = F.relu(self.layer1(obs))
        obs = F.relu(self.layer2(obs))

        mean = self.mean_layer(obs)
        log_std = self.log_std_layer(obs)
        torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        dist = Normal(mean, std)

        if deterministic:
            action = mean
        else:
            action = dist.rsample()

        if with_logprob:
            logp_prob = dist.log_prob(action).sum(axis=-1)
            logp_prob -= (2*(np.log(2) - action - F.softplus(-2*action))).sum(axis=1)
        else:
            logp_prob = None


        action = torch.tanh(action)
        action = self.act_limit * action      

        return action , logp_prob.unsqueeze(-1)  
'''
    # Another Way to Compute Action
    def forward(self, obs, deterministic = False, with_logprob=True):
        obs = F.relu(self.layer1(obs))
        obs = F.relu(self.layer2(obs))

        mean = self.mean_layer(obs)
        log_std = self.log_std_layer(obs)

        torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        dist = Normal(mean, std) # distribution

        sample_action = dist.rsample()
        tanh_action = torch.tanh(action)
        actual_action = tanh_action * self.action_scale + self.action_bias  # sample action,  squeeze into [-1, 1], scale back actual_action

        log_prob = dist.log_prob(sample_action)
        log_prob -= torch.log(self.action_scale * (1 - tanh_action.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean
'''

class GaussianPolicyMLPCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size, act_limit):
        super().__init__()
        self.policy = GaussianPolicy(obs_dim, act_dim, hidden_size, act_limit)
        self.critic = TwinQNetwork(obs_dim, act_dim, hidden_size)







    
