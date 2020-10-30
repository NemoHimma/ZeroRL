import torch
import numpy as np
from networks.sac_model import GaussianPolicyMLPCritic
import torch.optim as optim
import torch.nn.functional as F
'''
declare_networks()
declare_optimizer()
prepare_minibatch()
compute_value_loss()
compute_policy_loss()
update()
'''

class SACAgent(object):
    def __init__(self, obs_dim, act_dim, act_limit, args, device, writer=None):
        super().__init__()
        # get the variable
        self.args = args
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.writer = writer

        self.declare_networks()
        self.declare_optimizers()

    def declare_networks(self):
        self.model = GaussianPolicyMLPCritic(self.obs_dim, self.act_dim, self.args.hidden_size, self.act_limit)
        self.target_model = GaussianPolicyMLPCritic(self.obs_dim, self.act_dim, self.args.hidden_size, self.act_limit)
        self.target_model.load_state_dict(self.model.state_dict()) # hard update initial
        for param in self.target_model.parameters():
            param.requires_grad = False
        self.model = self.model.to(self.device)
        self.target_model = self.target_model.to(self.device)

    def declare_optimizers(self):
        self.policy_optimizer = optim.Adam(self.model.policy.parameters(), lr = self.args.policy_lr)
        self.critic_optimizer = optim.Adam(self.model.critic.parameters(), lr = self.args.value_lr)

    def get_action(self, obs, deterministic = False):
        '''
        input: obs is numpy.array with shape (obs_dim, ), transfer into (1, obs_dim)
        output: action is tensor with shape (1, obs_dim), trasfer into numpy.array with shape (act_dim, )
        '''
        obs = torch.as_tensor(obs, dtype = torch.float32, device= self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.model.policy(obs, deterministic)
        action = action.detach().cpu().numpy()[0]
        return action    

    def prepare_minibatch(self, buffer):
        idxs = np.random.randint(0, buffer.size, self.args.batch_size)
        obs_batch = torch.as_tensor(buffer.obs_buf[idxs], dtype = torch.float32, device = self.device)
        acts_batch = torch.as_tensor(buffer.acts_buf[idxs], dtype = torch.float32, device = self.device)
        rews_batch = torch.as_tensor(buffer.rew_buf[idxs], dtype = torch.float32, device= self.device).unsqueeze(-1)
        next_obs_batch = torch.as_tensor(buffer.next_obs_buf[idxs], dtype = torch.float32, device= self.device)
        done_batch = torch.as_tensor(buffer.done_buf[idxs], dtype = torch.float32, device= self.device).unsqueeze(-1)

        return obs_batch, acts_batch, rews_batch, next_obs_batch, done_batch


    def compute_value_loss(self, batch_data):
        obs, act, rew, next_obs, done = batch_data
        q1_value, q2_value = self.model.critic(obs, act)

        with torch.no_grad():
            next_acts, next_probs = self.model.policy(next_obs)
            
            target_q1_value , target_q2_value = self.target_model.critic(next_obs, next_acts)
            
            
            target_q_value = torch.min(target_q1_value, target_q2_value)
            
           
            target_update = rew + self.args.gamma * (1 - done) * (target_q_value - self.args.alpha * next_probs)
            

        loss1 = F.mse_loss(q1_value, target_update)
        loss2 = F.mse_loss(q2_value, target_update)
        value_loss = loss1 + loss2

        return value_loss

    def compute_policy_loss(self, batch_data):
        obs, _, _, _, _ = batch_data
        act, log_prob = self.model.policy(obs)
        q1_value, q2_value = self.model.critic(obs, act)
        q_value = torch.min(q1_value, q2_value)

        policy_loss = (-q_value + self.args.alpha * log_prob).mean()
        
        return policy_loss

    def update(self, buffer, step):
        value_loss_log = []
        policy_loss_log = []
        
        for i in range(self.args.updates_per_step):
            batch_data = self.prepare_minibatch(buffer)


            value_loss = self.compute_value_loss(batch_data)
            value_loss_log.append(value_loss.detach().cpu().numpy())

            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()
        

            if (i + 1) % self.args.policy_delay == 0:
                policy_loss = self.compute_policy_loss(batch_data)
                policy_loss_log.append(policy_loss.detach().cpu().numpy())
                

                for param in self.model.critic.parameters():
                    param.requires_grad = False
                
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                for param in self.model.critic.parameters():
                    param.requires_grad = True

            if (i + 1) % self.args.target_update_freq == 0:
                with torch.no_grad():
                    for param, target_parma in zip(self.model.parameters(), self.target_model.parameters()):
                        target_parma.data.mul_(self.args.polyak)
                        target_parma.data.add_((1-self.args.polyak) * param.data)
        
        return np.mean(value_loss_log), np.mean(policy_loss_log)




            
        