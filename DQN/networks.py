import torch
import torch.nn as nn
import torch.nn.functional as F

from bodies import SimpleBody , AtariBody
from noisy_layer import NoisyLinear

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions, noisy=False, sigma_init=0.5, body=SimpleBody):
        super(DQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.noisy = noisy

        self.body = body(self.input_shape, self.num_actions, self.noisy)
        
        if not self.noisy:
            self.fc1 = nn.Linear(self.body.feature_size(), 512)
            self.fc2 = nn.Linear(512, self.num_actions)
        else:
            self.fc1 = NoisyLinear(self.body.feature_size(), 512, sigma_init)
            self.fc2 = NoisyLinear(512, self.num_actions, sigma_init)

    def forward(self, x):
        x = self.body(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x # [batch_size, num_actions]

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.fc1.sample_noise()
            self.fc2.sample_noise()


class DuelingDQN(nn.Module):
    def __init__(self, input_shape, num_actions, noisy=False, sigma_init=0.5, body=SimpleBody):
        super(DuelingDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.noisy = noisy

        self.body = body(self.input_shape, self.num_actions, noisy, sigma_init)

        if not self.noisy:
            self.adv1 = nn.Linear(self.body.feature_size(), 512)
            self.adv2 = nn.Linear(512, self.num_actions)

            self.val1 = nn.Linear(self.body.feature_size(), 512)
            self.val2 = nn.Linear(512, 1)
        else:
            self.adv1 = nn.NoisyLinear(self.body.feature_size(), 512, sigma_init)
            self.adv2 = nn.NoisyLinear(512, self.num_actions, sigma_init)

            self.val1 = nn.NoisyLinear(self.body.feature_size(), 512,sigma_init)
            self.val2 = nn.NoisyLinear(512, 1, sigma_init)

    def forward(self, x):
        x = self.body(x)
        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv)

        val = F.relu(self.val1(x))
        val = self.val2(val)

        return val + adv - adv.mean() # [batch_size, num_actions]

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.adv1.sample_noise()
            self.adv2.sample_noise()
            self.val1.sample_noise()
            self.val2.sample_noise()


class CategoricalDQN(nn.Module):
    def __init__(self, input_shape, num_actions, noisy=False, atoms=51, sigma_init=0.5, body=SimpleBody):
        super(CategoricalDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.noisy = noisy
        self.atoms = atoms

        self.body = body(self.input_shape, self.num_actions, self.noisy, sigma_init)

        if not self.noisy:
            self.fc1 = nn.Linear(self.body.feature_size(), 512)
            self.fc2 = nn.Linear(512, self.num_actions*self.atoms)
        else:
            self.fc1 = NoisyLinear(self.body.feature_size(), 512, sigma_init)
            self.fc2 = NoisyLinear(512, self.num_actions*self.atoms, sigma_init)

    def forward(self, x):
        x = self.body(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.softmax(x.view(-1, self.num_actions, self.atoms), dim=2) # [batch_size, num_actions, atoms]
    
    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.fc1.sample_noise()
            self.fc2.sample_noise()

class CategoricalDuelingDQN(nn.Module):
    def __init__(self, input_shape, num_actions, noisy=False, atoms=51, sigma_init=0.5, body=SimpleBody):
        super(CategoricalDuelingDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.noisy = noisy
        self.atoms = atoms

        self.body = body(self.input_shape, self.num_actions, self.noisy, sigma_init)

        if not self.noisy:
            self.adv1 = nn.Linear(self.body.feature_size(), 512)
            self.adv2 = nn.Linear(512, self.num_actions*self.atoms)

            self.val1 = nn.Linear(self.body.feature_size(), 512)
            self.val2 = nn.Linear(512, self.atoms)
        else:
            self.adv1 = NoisyLinear(self.body.feature_size(), 512, sigma_init)
            self.adv2 = NoisyLinear(512, self.num_actions*self.atoms, sigma_init)

            self.val1 = NoisyLinear(self.body.feature_size(), 512, sigma_init)
            self.val2 = NoisyLienar(512, self.atoms,sigma_init)

    def forward(self, x):
        x = self.body(x)

        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv).view(-1, self.num_actions, self.atoms)

        val = F.relu(self.val1(x))
        val = self.val2(val).view(-1, 1, self.atoms)

        final = val + adv - adv.mean(dim=1).view(-1, 1, self.atoms)

        return F.softmax(final, dim=2) 

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.adv1.sample_noise()
            self.adv2.sample_noise()
            self.val1.sample_noise()
            self.val2.sample_noise()



class DuelingQRDQN(nn.Module):
    def __init__(self, input_shape, num_actions, noisy=False, quantiles=51, sigma_init=0.5, body=SimpleBody):
        super(CategoricalDuelingDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.noisy = noisy
        self.quantiles = quantiles

        self.body = body(self.input_shape, self.num_actions, self.noisy, sigma_init)

        if not self.noisy:
            self.adv1 = nn.Linear(self.body.feature_size(), 512)
            self.adv2 = nn.Linear(512, self.num_actions*self.quantiles)

            self.val1 = nn.Linear(self.body.feature_size(), 512)
            self.val2 = nn.Linear(512, self.quantiles)
        else:
            self.adv1 = NoisyLinear(self.body.feature_size(), 512, sigma_init)
            self.adv2 = NoisyLinear(512, self.num_actions*self.quantiles, sigma_init)

            self.val1 = NoisyLinear(self.body.feature_size(), 512, sigma_init)
            self.val2 = NoisyLienar(512, self.quantiles,sigma_init)

    def forward(self, x):
        x = self.body(x)

        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv).view(-1, self.num_actions, self.quantiles)

        val = F.relu(self.val1(x))
        val = self.val2(val).view(-1, 1, self.quantiles)

        final = val + adv - adv.mean(dim=1).view(-1, 1, self.quantiles)

        return final  # [batch_size, num_actions, quantiles]

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.adv1.sample_noise()
            self.adv2.sample_noise()
            self.val1.sample_noise()
            self.val2.sample_noise()