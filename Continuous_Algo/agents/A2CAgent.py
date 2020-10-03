import torch
import torch.optim as optim
import numpy as np

from networks.networks import ActorCritic
from networks.bodies import MLPBody


# All the Agents shared RolloutBuffer in Parallel Setting so it doesn't have the declare_memory method

class A2CAgent(object):
    def __init__(self, config=None, envs=None):

        self.envs = envs
        self.obs_shape = self.envs.observation_space.shape
        self.action_space = self.envs.action_space
        self.num_steps = config.num_steps
        self.num_processes = config.num_processes

        self.lr = config.lr
        self.eps = config.eps
        self.alpha = config.alpha

        self.value_loss_coef = config.value_loss_coef
        self.entropy_loss_coef = config.entropy_loss_coef
        self.max_grad_norm = config.max_grad_norm

        # declare_networks()
        self.actor_critic_model = ActorCritic(obs_shape=self.input_shape, action_space=self.action_space, body=MLPBody)
        self.actor_critic_model.to(self.device)

        # declare_optimizer()
        self.optimizer = optim.RMSprop(actor_critic_model.parameters(), self.lr, eps=self.eps, alpha=self.alpha)


################# Main Update Step #################
    def update(self, rollouts):

        loss, value_loss, action_loss, dist_entropy = self.compute_loss(rollouts)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.actor_critic_model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()





######################################################
    def compute_loss(self, rollouts):
        values, action_log_probs, dist_entropy = self.evaluate_actions(rollouts.obs[:-1].view(-1, *self.obs_shape), rollouts.masks[:-1].view(-1, 1), rollouts.actions.view(-1, self.action_space))

        values = values.view(self.num_steps, self.num_processes, 1)
        action_log_probs = action_log_probs.view(self.num_steps, self.num_processes, 1)

        adv = rollouts.returns[:-1] - values

        value_loss = adv.pow(2).mean()
        action_loss = -(adv.detach()*action_log_probs).mean()

        loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_loss_coef

        return loss, value_loss, action_loss, dist_entropy



    def evaluate_actions(self, inputs, actions):
        critic_value, dist = self.actor_critic_model(inputs)

        action_log_probs = dist.log_probs(actions)
        dist_entropy = dist.entropy().mean()

        return critic_value, action_log_probs, dist_entropy


    def get_critic_value(self, inputs):
        critic_value, _ = self.actor_critic_model(inputs)
        return critic_value



    def get_action(self, inputs, deterministic = False):
        critic_value, dist = self.actor_critic_model(inputs)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        
        return critic_value, action, action_log_probs















