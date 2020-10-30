import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from buffer.ddpgBuffer import DDPGReplayBuffer
from networks.sac_model import GaussianActorMLPCritic
from agents.TD3Agent import TD3Agent
from tensorboardX import SummaryWriter

import itertools


'''
All the agent have methods:
update() : prep_minibatch() , get_action(), compute_policy_loss(), compute_value_loss()
create : declare_networks(), declare_memory(), declare_optimizers()
'''

class SACAgent(TD3Agent):
    # No need to rewrite prep_minibatch(), declare_memory()
    def __init__(self, config, env):
        super(SACAgent, self).__init__(config, env)
        self.alpha = 0.02
        self.policy_decay = 2
        self.log_dir = './results/sac_exp_reward/'
        self.writer = SummaryWriter(self.log_dir)
        self.critic_optimizer = optim.Adam(itertools.chain(self.model.critic1.parameters(), self.model.critic2.parameters()), lr = self.value_lr)

    def declare_networks(self):
        self.model = GaussianActorMLPCritic(self.observation_space, self.action_space)
        self.target_model = GaussianActorMLPCritic(self.observation_space, self.action_space)


    def get_action(self, obs, deterministic = False):
        # deterministic = True means Test the model
        obs = torch.as_tensor(obs).to(self.device)
        with torch.no_grad():
            action, _ = self.model.actor(obs, deterministic, False) # not calculate the action part (with_prob = False)
        action = action.cpu().numpy()
        return action

    def prepare_minibatch(self, batch_size):
        idxs = np.random.randint(0, self.buffer.size, batch_size)
        obs_batch = torch.as_tensor(self.buffer.obs_buf[idxs]).to(self.device)
        next_obs_batch = torch.as_tensor(self.buffer.next_obs_buf[idxs]).to(self.device)
        act_batch = torch.as_tensor(self.buffer.acts_buf[idxs]).to(self.device)
        rew_batch = torch.as_tensor(self.buffer.rew_buf[idxs]).to(self.device)
        done_batch = torch.as_tensor(self.buffer.done_buf[idxs]).to(self.device)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_batch

    def compute_policy_loss(self, batch_data, timestep):
        obs, _, _, _ ,_ = batch_data
        action , log_prob = self.model.actor(obs) # log_prob is entropy-regularised term 
        q_value1 = self.model.critic1(obs, action)
        q_value2 = self.model.critic2(obs, action)
        q_value = torch.min(q_value1, q_value2)
        self.writer.add_scalar('mean_log_prob', log_prob.detach().mean().cpu().numpy(), timestep)
        self.writer.add_scalar('mean_q_value_policy', q_value.detach().mean().cpu().numpy(), timestep)

        #policy_loss = (-q_value + self.alpha * log_prob).mean()
        policy_loss = (-q_value + self.alpha * log_prob).mean()
        #self.writer.add_scalar('step_policy_loss', policy_loss.detach().cpu().numpy(), timestep)

        return policy_loss

    def compute_value_loss(self, batch_data, timestep):
        obs, acts, rews, next_obs, dones = batch_data

        current_q_value1 = self.model.critic1(obs, acts)
        current_q_value2 = self.model.critic2(obs, acts)

        with torch.no_grad():
            next_acts , next_log_probs = self.model.actor(next_obs)

            # Target_q
            target_q_value1 = self.target_model.critic1(next_obs, next_acts)
            target_q_value2 = self.target_model.critic2(next_obs, next_acts)
            target_q_value = torch.min(target_q_value1, target_q_value2)
            #target_update = rews + self.gamma * (1 - dones) * (target_q_value - self.alpha * next_log_probs)
            target_update = rews + self.gamma * (1 - dones) * target_q_value
        
        loss1 = F.mse_loss(current_q_value1, target_update)
        loss2 = F.mse_loss(current_q_value2, target_update)

        loss = loss1 + loss2
        self.writer.add_scalar('step_value_loss', loss.detach().cpu().numpy(), timestep)
        
        return loss
    
    def update(self, update_times, timestep):
        value_log = []
        policy_log = []

        for i in range(update_times):
            batch_data = self.prepare_minibatch(self.batch_size)

            value_loss = self.compute_value_loss(batch_data, timestep)

            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()


            value_log.append(value_loss.detach().cpu().numpy())

            if (i+1) % self.policy_decay == 0:
            # Compute Policy Loss Use Critic Network so it requires frozen critic parameters
                for param in self.model.critic1.parameters():
                    param.requires_grad = False
                for param in self.model.critic2.parameters():
                    param.requires_grad = False

                self.actor_optimizer.zero_grad()
                policy_loss = self.compute_policy_loss(batch_data, timestep) 
                policy_loss.backward()
                self.actor_optimizer.step()

                policy_log.append(policy_loss.detach().cpu().numpy())


                for param in self.model.critic1.parameters():
                    param.requires_grad = True
                
                for param in self.model.critic2.parameters():
                    param.requires_grad = True

                # soft update
                with torch.no_grad():
                    for param, target_parma in zip(self.model.parameters(), self.target_model.parameters()):
                        target_parma.data.mul_(self.polyak)
                        target_parma.data.add_((1-self.polyak) * param.data)
        
        return np.mean(value_log), np.mean(policy_log)    


        

