import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from buffer.ddpgBuffer import DDPGReplayBuffer
from networks.sac_model import GaussianActorMLPCritic

import itertools


'''
All the agent have methods:
update() : prep_minibatch() , get_action(), compute_policy_loss(), compute_value_loss()
create : declare_networks(), declare_memory(), declare_optimizers()
'''

class SACAgent(object):
    # No need to rewrite prep_minibatch(), declare_memory()
    def __init__(self, args, env, log_dir, device):
        
        # args
        self.alpha = args.alpha 
        self.policy_decay = args.policy_decay
        self.value_lr = args.value_lr
        self.policy_lr = args.policy_lr
        self.polyak = args.polyak
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.target_update_freq = args.target_update_freq


        # declare_networks & declare_memory

        self.net_kwargs = {'actor_hidden_size':args.actor_hidden_size,
        'critic_hidden_size':args.critic_hidden_size,
        'actor_log_std_min':args.actor_log_std_min,
        'actor_log_std_max':args.actor_log_std_max}

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.act_limit = env.action_space.high[0]
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.buffer_size = args.buffer_size

        # log_dir & device
        self.log_dir = log_dir
        self.device = device

        self.declare_networks()
        self.declare_memory()

        self.target_model.load_state_dict(self.model.state_dict())

        # declare_optimizer
        self.critic_optimizer = optim.Adam(itertools.chain(self.model.critic1.parameters(), self.model.critic2.parameters()), lr = self.value_lr)
        self.actor_optimizer = optim.Adam(self.model.actor.parameters(), lr = self.policy_lr)

        self.model = self.model.to(self.device)
        self.target_model = self.target_model.to(self.device)
        
        # Freeze Target
        for param in self.target_model.parameters():
            param.requires_grad = False

        # auto_tune_alpha
        self.auto_tune_alpha = args.auto_tune_alpha
        if self.auto_tune_alpha:
            self.target_entropy = -torch.prod(torch.Tensor(self.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=args.alpha_lr)           
       
            
            

    def declare_networks(self):
        self.model = GaussianActorMLPCritic(self.observation_space, self.action_space, **self.net_kwargs)
        self.target_model = GaussianActorMLPCritic(self.observation_space, self.action_space, **self.net_kwargs)
    
    def declare_memory(self):
        self.buffer = DDPGReplayBuffer(self.obs_dim, self.act_dim, self.buffer_size)


    def get_action(self, obs, deterministic = False):
        # deterministic = True means Test the model
        obs = torch.as_tensor(obs, dtype = torch.float32).to(self.device)
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
        
       

        #policy_loss = (-q_value).mean()
        policy_loss = (-q_value + self.alpha * log_prob).mean()
        #self.writer.add_scalar('step_policy_loss', policy_loss.detach().cpu().numpy(), timestep)

        return policy_loss, log_prob, q_value.detach().mean().cpu().numpy()

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
            target_update = rews + self.gamma * (1-dones) * (target_q_value - self.alpha * next_log_probs)
            #target_update = rews + self.gamma * (1 - dones) * target_q_value
        
        loss1 = F.smooth_l1_loss(current_q_value1, target_update)
        loss2 = F.smooth_l1_loss(current_q_value2, target_update)

        loss = loss1 + loss2
        
        
        return loss
    
    def update(self, update_times, totalstep):
        value_log = []
        policy_log = []
        log_prob_log = []
        q_value_log = []
        #alpha_log = []
        

        for i in range(update_times):
            batch_data = self.prepare_minibatch(self.batch_size)

            value_loss = self.compute_value_loss(batch_data, totalstep)

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
                policy_loss, log_prob, q_value = self.compute_policy_loss(batch_data, totalstep) 
                policy_loss.backward()
                self.actor_optimizer.step()

                policy_log.append(policy_loss.detach().cpu().numpy())
                log_prob_log.append(log_prob.detach().mean().cpu().numpy())
                q_value_log.append(q_value)


                for param in self.model.critic1.parameters():
                    param.requires_grad = True
                
                for param in self.model.critic2.parameters():
                    param.requires_grad = True

                # update alpha
                if self.auto_tune_alpha:
                    alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()

                    self.alpha = self.log_alpha.exp()
                    alpha_to_log = self.alpha.clone()
                    alpha_log.append(alpha_to_log.detach().cpu().numpy())

                # soft update
                if totalstep % self.target_update_freq == 0:
                    with torch.no_grad():
                        for param, target_parma in zip(self.model.parameters(), self.target_model.parameters()):
                            target_parma.data.mul_(self.polyak)
                            target_parma.data.add_((1-self.polyak) * param.data)
        
        return np.mean(value_log), np.mean(policy_log),np.mean(log_prob_log), np.mean(q_value_log)#,np.mean(alpha_log)


        

