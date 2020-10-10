import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class RolloutBuffer(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space):
        # (obs, act, reward)
        self.obs = torch.zeros(num_steps+1, num_processes, *obs_shape)   # (obs[0] = init_obs)

        ## Action is Discrete or Box:
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.acts = torch.zeros(num_steps, num_processes, action_shape) 

        if action_space.__class__.__name__ == 'Discrete':
            self.acts = self.acts.long()

        self.rewards = torch.zeros(num_steps, num_processes, 1)

        # ( logpi(a|s), V(s), G )
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps+1, num_processes, 1) # value_preds[-1] = next_value which deals with truncated episode (bad transition)
        self.returns = torch.zeros(num_steps+1, num_processes, 1) # Return size correspond to value_pred

        # mask 0 for natural teminal obs & bad_mask 0 for bad transition obs
        self.masks = torch.ones(num_steps+1, num_processes, 1)
        self.bad_masks = torch.ones(num_steps+1, num_processes, 1) # proper_time_limits

        # utility
        self.num_steps = num_steps
        self.pointer = 0 # store element into buffer

    # Let the Interaction  all base on Tensor data type
    def to_device(self, device):
        self.obs = self.obs.to(device)
        self.acts = self.acts.to(device)
        
        self.rewards = self.rewards.to(device)

        self.action_log_probs = self.action_log_probs.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)


    def store(self, next_obs, acts, rewards, action_log_probs, value_preds, masks, bad_masks):

        ''' store (next_obs, masks, bad_masks) from next_obs & (acts, rewards, action_log_probs, value_preds) from current_obs '''

        self.obs[self.pointer+1].copy_(next_obs) 

        self.value_preds[self.pointer].copy_(value_preds)

        self.acts[self.pointer].copy_(acts) 
        self.action_log_probs[self.pointer].copy_(action_log_probs)
        self.rewards[self.pointer].copy_(rewards)

        self.masks[self.pointer+1].copy_(masks)
        self.bad_masks[self.pointer+1].copy_(bad_masks)

        self.pointer = (self.pointer + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    # default to use gae...  'proper_time_limits = True'
    def compute_returns(self, next_value, gamma, gae_lambda, proper_time_limits= True):
        self.value_preds[-1] = next_value
        gae = 0
        # gae_t = \sum_{l=0}^{\infty} (\lambda \gamma)^l \delta_{t}  \lambda summing up and discount multi-step advantage estimator
        if proper_time_limits == True:
                 
            for step in reversed(range(self.num_steps)): # index over num_steps-1,...,0
                # \delat_t = r_t + \gamma * V(s_t) - V(s_{t-1})
                delta_step = self.rewards[step] + gamma * self.value_preds[step+1] * self.masks[step+1] - self.value_preds[step] 
                gae = delta_step + gamma * gae_lambda * gae * self.masks[step+1]
                gae = gae * self.bad_masks[step + 1]  # If next step is bad transition,  proper_time_limits = True
                self.returns[step]  = gae + self.value_preds[step]

        else:

            for step in reversed(range(self.num_steps)):
                delta_step = self.rewards[step] + gamma * self.value_preds[step+1] * self.masks[step+1] - self.value_preds[step]
                gae = delta_step + gamma * gae_lambda * gae * self.masks[step+1]

                self.returns[step] = gae + self.value_preds[step]












