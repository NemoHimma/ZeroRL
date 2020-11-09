import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal


# Initialize Linear Layer weights & bias
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1.414)
        torch.nn.init.constant_(m.bias, 0)

class TanhGaussianActor(nn.Module):

    def __init__(self, observation_space, action_space, actor_hidden_size, actor_log_std_min, actor_log_std_max):
        super(TanhGaussianActor, self).__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]

        self.actor_log_std_min = actor_log_std_min
        self.actor_log_std_max = actor_log_std_max

        
        self.actor_layer1 = nn.Linear(obs_dim, actor_hidden_size)
        self.actor_layer2 = nn.Linear(actor_hidden_size, actor_hidden_size)
        self.mu_layer = nn.Linear(actor_hidden_size, act_dim)
        self.log_std_layer = nn.Linear(actor_hidden_size, act_dim)

        self.apply(weights_init_)

    def forward(self, obs, deterministic = False, with_logprob = True): # Actor

        actor_tmp = F.relu(self.actor_layer1(obs))
        actor_tmp = F.relu(self.actor_layer2(actor_tmp))
        mu = self.mu_layer(actor_tmp)
        log_std = self.log_std_layer(actor_tmp)
        log_std = torch.clamp(log_std, self.actor_log_std_min, self.actor_log_std_max)
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

    def __init__(self, observation_space, action_space, critic_hidden_size):
        super(MLPCritic, self).__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        self.q_value_layer1 = nn.Linear(obs_dim+act_dim, 256)

        #print("layer:{0}".format(self.q_value_layer1.weight.requires_grad))
        
        self.q_value_layer2 = nn.Linear(critic_hidden_size, critic_hidden_size)
        self.q_value = nn.Linear(critic_hidden_size, 1)

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
    def __init__(self, observation_space, action_space,actor_hidden_size, critic_hidden_size, actor_log_std_min, actor_log_std_max):
        super(GaussianActorMLPCritic, self).__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]

        self.actor = TanhGaussianActor(observation_space, action_space, actor_hidden_size, actor_log_std_min, actor_log_std_max)
        self.critic1 = MLPCritic(observation_space, action_space, critic_hidden_size)
        self.critic2 = MLPCritic(observation_space, action_space, critic_hidden_size)

    




    


