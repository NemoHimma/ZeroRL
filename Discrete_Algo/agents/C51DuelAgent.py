import torch
import pickle
import torch.optim as optim
import numpy as np
import csv
import os 

from agents.DQNAgent import DQNAgent
from buffer.ReplayBuffer import ExperienceReplayBuffer
from networks.networks import CategoricalDuelingDQN
from networks.bodies import AtariBody

# Categorical, Dueling, N-Steps
class C51DuelAgent(DQNAgent):
    def __init__(self, config, env, log_dir, static_policy=False):
        super(C51DuelAgent, self).__init__(config, env, log_dir, static_policy=False)

        self.atoms = config.ATOMS
        self.v_max = config.V_MAX
        self.v_min = config.V_MIN

        # fix the location
        self.supports = torch.linspace(self.v_min, self.v_max, self.atoms).view(1, 1, self.atoms).to(config.device)
        self.delta = (self.v_max - self.v_min) / (self.atoms - 1)

    def declare_networks(self):
        self.model = CategoricalDuelingDQN(self.input_shape, self.num_actions, self.noisy, self.atoms, self.sigma_init, body = AtariBody)
        self.target_model = CategoricalDuelingDQN(self.input_shape, self.num_actions, self.noisy, self.atoms, self.sigma_init, body = AtariBody)

    def projection_distribution(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        with torch.no_grad():
            max_next_dist = torch.zeros((self.batch_size, 1, self.atoms), device=self.device, dtype=torch.float) + 1. / self.atoms
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                self.target_model.sample_noise()
                max_next_dist[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_action)
                max_next_dist = max_next_dist.squeeze()

            Tz = batch_reward.view(-1, 1) + (self.gamma**self.nsteps) * self.supports.view(1, -1) * non_final_mask.to(torch.float).view(-1, 1)
            Tz = Tz.clamp(self.v_min, self.v_max)
            b = (Tz - self.v_min) / self.delta
            l = b.floor().to(torch.int64)
            u = b.ceil().to(torch.int64)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1
            
            offset = torch.linspace(0, (self.batch_size - 1) * self.atoms, self.batch_size).unsqueeze(dim=1).expand(self.batch_size, self.atoms).to(batch_action)
            m = batch_state.new_zeros(self.batch_size, self.atoms)
            m.view(-1).index_add_(0, (l + offset).view(-1), (max_next_dist * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1), (max_next_dist * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)
        
        return m

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        batch_action = batch_action.unsqueeze(dim=-1).expand(-1, -1, self.atoms)
        batch_reward = batch_reward.view(-1, 1, 1)

        # estimate
        self.model.sample_noise()
        current_dist = self.model(batch_state).gather(1, batch_action).squeeze()

        target_prob = self.projection_distribution(batch_vars)
          
        loss = -(target_prob * current_dist.log()).sum(-1)
        if self.priority_replay:
            self.memory.update_priorities(indices, loss.detach().squeeze().abs().cpu().numpy().tolist())
            loss = loss * weights
        loss = loss.mean()

        return loss

    def get_action(self, s, eps):
        with torch.no_grad():
            if np.random.random() >= eps or self.static_policy or self.noisy:
                X = torch.tensor([s], device=self.device, dtype=torch.float) 
                self.model.sample_noise()
                a = self.model(X) * self.supports
                a = a.sum(dim=2).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

# action selection from target_model which is not double
    def get_max_next_state_action(self, next_states):
        next_dist = self.target_model(next_states) * self.supports
        return next_dist.sum(dim=2).max(1)[1].view(next_states.size(0), 1, 1).expand(-1, -1, self.atoms)



