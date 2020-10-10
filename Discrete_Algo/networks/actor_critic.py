import torch
import torch.nn as nn
import torch.nn.functional as F




class CNNActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CNNActorCritic, self).__init__()

        # Initialize conv layer weight orthogonally with relu gain and bias 0
        init_ = lambda m: self.init(m , nn.init.orthogonal_, lambda x : nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))

        self.conv1 = init_(nn.Conv2d(input_shape[0], 32, 8, stride=4))
        self.conv2 = init_(nn.Conv2d(32, 64, 4, stride=2))
        self.conv3 = init_(nn.Conv2d(64, 32, 3, stride=1))

        # extract image features
        self.fc1 = init_(nn.Linear(self.feature_size(input_shape), 512)) 
        
        # Initialize actor&critic differently
        init_critic = lambda m : self.init(m, nn.init.orthogonal_, lambda x:nn.init.constant_(x, 0))
        init_actor = lambda m : self.init(m, nn.init.orthogonal_, lambda x : nn.init.constant_(x, 0), gain=0.01)

        self.critic_value = init_critic(nn.Linear(512, 1))
        self.actor_features = init_actor(nn.Linear(512, num_actions))

        

        self.train()

    def forward(self, x):
        x = F.relu(self.conv1(x/255.0)) # Remember to Normalize the inputs
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # (batch_size, feature_size)
        x = F.relu(self.fc1(x))

        value = self.critic_value(x)
        logits = self.actor_features(x)

        return value, logits


    def init(self, module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module    

    def feature_size(self, input_shape):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *input_shape)))).view(1,-1).size(1)

        


