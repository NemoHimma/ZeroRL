import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPActor(nn.Module):

    def __init__(self, observation_space, action_space):
        super(MLPActor, self).__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]

        
        self.actor_layer1 = nn.Linear(obs_dim, 256)
        self.actor_layer2 = nn.Linear(256, 256)
        self.actor_features = nn.Linear(256, act_dim)

    def forward(self, obs): # Actorc
        actor_tmp = F.relu(self.actor_layer1(obs))
        actor_tmp = F.relu(self.actor_layer2(actor_tmp))
        actor_tmp = torch.tanh(self.actor_features(actor_tmp)) 

        return actor_tmp * self.act_limit       



class MLPCritic(nn.Module):

    def __init__(self, observation_space, action_space):
        super(MLPCritic, self).__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        self.q_value_layer1 = nn.Linear(obs_dim+act_dim, 256)

        #print("layer:{0}".format(self.q_value_layer1.weight.requires_grad))
        
        self.q_value_layer2 = nn.Linear(256, 256)
        self.q_value = nn.Linear(256, 1)

    def forward(self, obs, act):
        q = torch.cat([obs, act], dim=-1) #(batch_size, concated_obs_act)

        #print(self.q_value_layer1.weight.requires_grad)

        q = F.relu(self.q_value_layer1(q))

        #print("Critic Q value:{0}".format(q.requires_grad))

        q = F.relu(self.q_value_layer2(q))
        
        q_value = torch.squeeze(self.q_value(q), -1) # (batch_size, q_value)

        return q_value

# distangeled representation of actor & critic (or shared representation)
class MLPActorCritic(nn.Module):
    '''
    default:
    input = Box(obs_dim, )
    output = logits array

    '''
    def __init__(self, observation_space, action_space):
        super(MLPActorCritic, self).__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]

        self.actor = MLPActor(observation_space, action_space)
        self.critic1 = MLPCritic(observation_space, action_space)
        self.critic2 = MLPCritic(observation_space, action_space)

    




    


