import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from ..networks.dqn_model import DQN
from ..buffer.ReplayBuffer import ExperienceReplayBuffer

class DQNAgent(object):
    def __init__(self,args, env, log_dir, device):

        # args
        self.device = device         # device
        self.update_count = 0

        self.lr = args.lr                       # 1e-3
        self.gamma = args.gamma                 # 0.9
        self.batch_size = args.batch_size       # 64
        self.target_update_freq = args.target_update_freq   # 200
        
        # Exploration stragedy
        self.epsilon_start = args.epsilon_start             #  1.0
        self.epsilon_final = args.epsilon_final                 #  0.01
        self.epsilon_decay = args.epsilon_decay             #  1000
        self.epsilon_rate = args.epsilon_rate               # -0.2
        self.epsilon_by_step = lambda: totalstep: self.epsilon_final + (self.epsilon_start - self.epsilon_final)*math.exp(self.epsilon_rate * totalstep / self.epsilon_decay)
        
        # Entity Construction(buffer, model, optimizer) 
        # Initilization (model, optimizer)
        self.memory_size = args.memory_size     # 1e6

        self.env = env
        self.input_shape = env.observation_space.shape  # (6, )
        self.num_actions = env.action_space.n           # 13

        self.declare_networks()
        self.declare_memory()
        
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model = self.model.to(self.device)
        self.target_model = self.target_model.to(self.device)

    def get_action(self, obs, epsilon):
        
        with torch.no_grad():
            if np.random.random() >= epsilon or self.noisy:
                obs_tensor = torch.tensor([obs], device=self.device, dtype=torch.float)
                self.model.sample_noise()
                action = self.model(obs_tensor).max(1)[1].view(1, 1)
                return action.item()
            
            else:
                np.random.randint(0, self.num_actions)
            
    def prepare_minibatch(self):
        transitions = self.buffer.sample(self.batch_size)
        obs_batch, act_batch, rew_batch, next_obs_batch = zip(*transitions)

        batch_states_shape = (-1, ) + self.input_shape

        obs_batch = torch.tensor(obs_batch, device = self.device, dtype = torch.float).view(batch_states_shape)
        act_batch = torch.tensor(act_batch, device = self.device, dtype = torch.long).view(-1, 1)
        rew_batch = torch.tensor(rew_batch, device = self.device, dtype = torch.float).view(-1, 1)
        next_obs_batch = torch.tensor(next_obs_batch, device = self.device, dtype = torch.float).view(batch_states_shape)

        return obs_batch, act_batch, rew_batch, next_obs_batch

    def compute_loss(self, batch_data):
        obs_batch, act_batch, rew_batch, next_obs_batch = batch_data

        # current_q_value
        self.model.sample_noise()
        current_q_value = self.model(obs_batch).gather(1, act_batch)

        # target_value
        with torch.no_grad():
            next_max_action = self.target_model(next_obs_batch).max(dim=1)[1].view(-1, 1)
            self.target_model.sample_noise()
            next_max_q_value = self.target_model(next_obs_batch).gather(1, next_max_action)

        target_value = rew_batch + self.gamma * next_max_q_value

        loss = F.smooth_l1_loss(current_q_value, target_value)

        return loss


    def update(self):
        batch_data = self.prepare_minibatch()
        loss = self.compute_loss(batch_data)

        self.optimizer.zero_grad() 
        loss.backward()
        for parameter in self.model.parameters():
            parameter.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_target_model()
        
        return loss.item()

        
    def update_target_model(self):
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            self.update_count = 0        

    def declare_networks(self):
        self.model = DQN(self.input_shape, self.num_actions, args.noisy, args.sigma_init, args.hidden_size)
        self.target_model = DQN(self.input_shape, self.num_actions, args.noisy, args.sigma_init,args.hidden_size)
        
    def declare_memory(self):
        self.buffer = ExperienceReplayBuffer(self.memory_size)
        


if __name__ == '__main__':
    pass
