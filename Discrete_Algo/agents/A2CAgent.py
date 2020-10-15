import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from networks.actor_critic import CNNActorCritic
from buffer.rollout import RolloutBuffer


class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class A2CAgent(object):
    def __init__(self, config=None, envs=None):
        self.device = config.device

        self.envs = envs
        self.obs_shape = self.envs.observation_space.shape
        self.action_space = self.envs.action_space
        if self.action_space == 'Discrete':
            self.act_shape = 1



        self.num_steps = config.num_steps
        self.num_processes = config.num_processes
        

        self.lr = config.lr
        self.eps = config.eps
        self.alpha = config.alpha

        self.value_loss_coef = config.value_loss_coef
        self.entropy_loss_coef = config.entropy_loss_coef
        self.max_grad_norm = config.max_grad_norm

        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda

        # declare_networks()
        self.actor_critic_model = CNNActorCritic(input_shape=self.obs_shape, num_actions=self.action_space.n)
        self.actor_critic_model.to(self.device)

        # declare_memory()
        self.rollouts = RolloutBuffer(self.num_steps, self.num_processes, self.obs_shape, self.action_space)
        self.rollouts.to_device(self.device)

        # declare_optimizer()
        self.optimizer = optim.RMSprop(self.actor_critic_model.parameters(), self.lr, eps=self.eps, alpha=self.alpha)


################# Main Update Step #################
    def update(self):
        
        with torch.no_grad():
            next_value = self.get_critic_value(self.rollouts.obs[-1]).detach()

        self.rollouts.compute_returns(next_value, self.gamma, self.gae_lambda)
        
        loss, value_loss, action_loss, dist_entropy = self.compute_loss()
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.actor_critic_model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()





######################################################
    def compute_loss(self):
        values, action_log_probs, dist_entropy = self.prepare_minibatch()

        adv = self.rollouts.returns[:-1] - values

        value_loss = adv.pow(2).mean()
        action_loss = -(adv.detach()*action_log_probs).mean()

        loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_loss_coef

        return loss, value_loss, action_loss, dist_entropy

    def prepare_minibatch(self):
        values, action_log_probs, dist_entropy = self.evaluate_actions(self.rollouts.obs[:-1].view(-1, *self.obs_shape), self.rollouts.acts.view(-1, 1))

        values = values.view(self.num_steps, self.num_processes, 1)
        action_log_probs = action_log_probs.view(self.num_steps, self.num_processes, 1)

        return values, action_log_probs, dist_entropy

###################################


    def evaluate_actions(self, inputs, actions):
        critic_value, logits = self.actor_critic_model(inputs)
        
        dist = FixedCategorical(logits=logits)
        
        action_log_probs = dist.log_probs(actions)
        dist_entropy = dist.entropy().mean()

        return critic_value, action_log_probs, dist_entropy


    def get_critic_value(self, inputs):
        critic_value, _ = self.actor_critic_model(inputs)
        return critic_value



    def get_action(self, inputs, deterministic = False):
        critic_value, logits = self.actor_critic_model(inputs)
        
        dist = FixedCategorical(logits=logits)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        
        return critic_value, action, action_log_probs


################## Save & Log Info Part Remained To Do  #####################












