import torch
import torch.optim as optim
import numpy as np
import os

from networks.td3_model import MLPActorCritic
from buffer.ddpgBuffer import DDPGReplayBuffer

# distangeled representation of actor & critic (or shared representation)
class TD3Agent(object):
    def __init__(self, config=None, env=None):
        
        # get hyperparameters from config
        self.device = config.device
        self.buffer_size = config.buffer_size
        self.batch_size = config.batch_size
        self.policy_lr = config.policy_lr
        self.value_lr = config.value_lr
        self.gamma = config.gamma
        self.target_noise = config.target_noise
        
        self.polyak = config.polyak

        # Env Info
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.act_limit = env.action_space.high[0]

        ## Control Variable
        self.learn_start_steps = config.learn_start_steps
        self.update_freq = config.update_freq
        self.policy_decay = config.policy_decay

        # Construct Entities
        self.declare_networks()
        self.declare_memory()

        # Initilization & Optimizer
        
        self.target_model.load_state_dict(self.model.state_dict())

        self.actor_optimizer = optim.Adam(self.model.actor.parameters(), lr = self.policy_lr)
        self.critic1_optimizer = optim.Adam(self.model.critic1.parameters(), lr = self.value_lr)
        self.critic2_optimizer = optim.Adam(self.model.critic2.parameters(), lr = self.value_lr)

        self.model = self.model.to(self.device)
        self.target_model = self.target_model.to(self.device)


        # Freeze Target 
        for param in self.target_model.parameters():
            param.requires_grad = False
    


    def declare_networks(self):
        self.model = MLPActorCritic(self.observation_space, self.action_space)
        self.target_model = MLPActorCritic(self.observation_space, self.action_space)

    def declare_memory(self):
        self.buffer = DDPGReplayBuffer(self.obs_dim, self.act_dim, self.buffer_size)

##################### Key Update Part ##########################

    def update(self, update_times, target_noise):
        # if num_step < self.learn_start_steps:
        #    return None
        value_log = []
        policy_log = []
        for i in range(update_times):
            batch_data = self.prep_minibatch(self.batch_size)

            value1_loss, value2_loss = self.compute_value_loss(batch_data, target_noise)
            self.critic1_optimizer.zero_grad()
            value1_loss.backward()
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            value2_loss.backward()
            self.critic2_optimizer.step()

            value_log.append((value1_loss + value2_loss).item()/2)

            if (i+1) % self.policy_decay == 0:
            # Compute Policy Loss Use Critic Network so it requires frozen critic parameters
                for param in self.model.critic1.parameters():
                    param.requires_grad = False
                for param in self.model.critic2.parameters():
                    param.requires_grad = False

                self.actor_optimizer.zero_grad()
                policy_loss = self.compute_policy_loss(batch_data) 
                policy_loss.backward()
                self.actor_optimizer.step()

                policy_log.append(policy_loss.item())


                for param in self.model.critic1.parameters():
                    param.requires_grad = True
                
                for param in self.model.critic2.parameters():
                    param.requires_grad = True

                with torch.no_grad():
                    for param, target_parma in zip(self.model.parameters(), self.target_model.parameters()):
                        target_parma.data.mul_(self.polyak)
                        target_parma.data.add_((1-self.polyak) * param.data)
        
        return np.mean(value_log), np.mean(policy_log)



# Interaction
    def prep_minibatch(self, batch_size):

        '''
        This Part Can be written into Buffer as Method
        '''

        idxs = np.random.randint(0, self.buffer.size, batch_size)
        obs_batch = torch.from_numpy(self.buffer.obs_buf[idxs]).to(self.device)
        next_obs_batch = torch.from_numpy(self.buffer.next_obs_buf[idxs]).to(self.device)
        act_batch = torch.from_numpy(self.buffer.acts_buf[idxs]).to(self.device)
        rew_batch = torch.from_numpy(self.buffer.rew_buf[idxs]).to(self.device)
        # pay attention to done_type
        #done_batch = torch.from_numpy(self.buffer.done_buf[idxs]).to(self.device)

        return obs_batch, act_batch, rew_batch, next_obs_batch


    def get_action(self, obs , noisy_scale):
        '''
        Make Sure Input obs is numpy.array
        OutPut act is numpy.array
        '''
        obs = torch.from_numpy(obs).to(self.device)
        with torch.no_grad():
            a = self.model.actor(obs)
        
        a = a.cpu().numpy()
        # Add Noise when training But Test with noisy_scale = 0
        a += noisy_scale * np.random.randn(self.act_dim)

        return np.clip(a, -self.act_limit, self.act_limit)



# loss 
    def compute_value_loss(self, batch_data, target_noise):

        obs, acts, rews, next_obs = batch_data # batch tensor

        current_q_value1 = self.model.critic1(obs, acts)
        current_q_value2 = self.model.critic2(obs, acts)
        
       # print("current_q_value:{0},{1}".format(current_q_value, current_q_value.requires_grad))

       # print("obs_shape:{0}, acts_shape:{0}".format(obs.shape, acts.shape))

        # DDPG Style To choose action not a* = argmax_a Q(s,a)
        with torch.no_grad():
            next_acts = self.target_model.actor(next_obs)  # action selection from target_model which SAC use current model

            # Smoothing Targe Policy (Can Tune the hyperparameters)
            epsilon = torch.randn_like(next_acts) * target_noise
            epsilon = torch.clamp(epsilon, -self.act_limit, self.act_limit)  
            next_acts = torch.clamp(next_acts + epsilon , -self.act_limit, self.act_limit)


            target_q1 = self.target_model.critic1(next_obs, next_acts) # action evaluation from current model
            target_q2 = self.target_model.critic2(next_obs, next_acts)

            target_q_value = rews + self.gamma * torch.min(target_q1, target_q2)

       # print("target_q_value:{0}".format(target_q_value))

       # print("TD error:{0}".format((target_q_value - current_q_value).requires_grad))
        value_loss1 = ((target_q_value - current_q_value1)**2).mean()
        value_loss2 = ((target_q_value - current_q_value2)**2).mean()
        
        return value_loss1, value_loss2

    def compute_policy_loss(self, batch_data):
        obs, _, _, _ = batch_data
        q_value = self.model.critic1(obs, self.model.actor(obs))
        return -q_value.mean()







    
