import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal


LOG_STD_MIN = -20
LOG_STD_MAX = -4

# Initialize Linear Layer weights & bias
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1.414)
        torch.nn.init.constant_(m.bias, 0)

class TanhGaussianActor(nn.Module):

    def __init__(self, observation_space, action_space):
        super(TanhGaussianActor, self).__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]

        
        self.actor_layer1 = nn.Linear(obs_dim, 256)
        self.actor_layer2 = nn.Linear(256, 256)
        self.mu_layer = nn.Linear(256, act_dim)
        self.log_std_layer = nn.Linear(256, act_dim)

        self.apply(weights_init_)

    def forward(self, obs, deterministic = False, with_logprob = True): # Actor

        actor_tmp = F.relu(self.actor_layer1(obs))
        actor_tmp = F.relu(self.actor_layer2(actor_tmp))
        mu = self.mu_layer(actor_tmp)
        log_std = self.log_std_layer(actor_tmp)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        distribution = Normal(mu, std)
        if deterministic:
            action = mu
        else:
            action = distribution.rsample()

        # Reference for OpenAI SpinningUp's Implementation of SAC
        if with_logprob:
            logp_prob = distribution.log_prob(action).sum(axis=-1)
            logp_prob -= (2*(np.log(2) - action - F.softplus(-2*action))).sum(axis=1)
        else:
            logp_prob = None


        action = torch.tanh(action)
        action = self.act_limit * action


        return action, logp_prob       



class MLPCritic(nn.Module):

    def __init__(self, observation_space, action_space):
        super(MLPCritic, self).__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        self.q_value_layer1 = nn.Linear(obs_dim+act_dim, 256)

        #print("layer:{0}".format(self.q_value_layer1.weight.requires_grad))
        
        self.q_value_layer2 = nn.Linear(256, 256)
        self.q_value = nn.Linear(256, 1)

        self.apply(weights_init_)

    def forward(self, obs, act):
        q = torch.cat([obs, act], dim=-1) #(batch_size, concated_obs_act)

        #print(self.q_value_layer1.weight.requires_grad)

        q = F.relu(self.q_value_layer1(q))

        #print("Critic Q value:{0}".format(q.requires_grad))

        q = F.relu(self.q_value_layer2(q))
        
        q_value = torch.squeeze(self.q_value(q), -1) # (batch_size, q_value)

        return q_value

# distangeled representation of actor & critic (or shared representation)
class GaussianActorMLPCritic(nn.Module):
    '''
    default:
    input = Box(obs_dim, )
    output = logits array

    '''
    def __init__(self, observation_space, action_space):
        super(GaussianActorMLPCritic, self).__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]

        self.actor = TanhGaussianActor(observation_space, action_space)
        self.critic1 = MLPCritic(observation_space, action_space)
        self.critic2 = MLPCritic(observation_space, action_space)

    




    


