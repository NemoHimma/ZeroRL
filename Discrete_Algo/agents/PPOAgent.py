
import torch.optim as optim
from networks.actor_critic import CNNActorCritic
from buffer.rollouts import RolloutBuffer
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

# 
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions): #actions = (num_processes, 1)
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

# class PPOAgent(A2CAgent): also can inherit from A2CAgent 

class PPOAgent(object):

    def __init__(self, config, envs):
   '''
   declare_networks()
   declare_memory()
   declare_optimizer()
   '''        
        self.device = config.device

        self.num_steps = config.num_steps
        self.num_processes = config.num_processes

        self.obs_shape = envs.observation_space.shape
        self.action_space = envs.action_space

        if self.action_space == 'Discrete':
            self.act_shape = 1
        else:
            self.act_shape = self.action_space.shape[0]

        self.lr = config.lr
        self.eps = config.eps

        self.num_mini_batch = config.num_mini_batch

        

        # declare_memory()
        self.rollouts = RolloutBuffer(self.num_steps, self.num_processes, self.obs_shape, self.action_space)
        self.rollouts.to_device(self.device)

        # declare_networks()
        self.actor_critic_model = CNNActorCritic(self.obs_shape, self.action_space.n)
        self.actor_critic_model.to(self.device)

        # declare_optimizer()
        self.optimizer = optim.Adam(self.actor_critic_model.parameters(), lr=self.lr, eps=self.eps)

    ########### Main Update Step ###############
    '''
    Rollouts are finished.
    Remained to deals with one epoch sample = num_steps * num_processes 
    which requires data generator
    '''

    def update(self):

        with torch.no_grad():
            next_value = self.get_critic_value(self.rollouts.obs[-1]).detach()
            
        # Critical Part to Estimate Returns
        self.rollouts.compute_returns(next_value, self.gamma, self.gae_lambda)

        # Advanages = Returns - Values & Normalized

        rollouts_adv = self.rollouts.returns[:-1] - self.rollouts.values[:-1] # tensor

        rollouts_adv = (rollouts_adv - rollouts_adv.mean()) / (rollouts_adv.std() + 1e-5)


        # For one rollouts which means num_steps * num_processes , we perform multi-times ppo-update (update policy & value using different speed)

        data_generator = self.feed_forward_generator(rollouts_adv, self.num_mini_batch)

        for sample in data_generator:
            obs_batch, acts_batch, values_batch, values_batch, rewards_batch, returns_batch, masks_batch, old_action_log_probs_batch = sample

            # get_current_action_log_probs


            # policy_loss
            self.


            # value_loss 

            pass



      

        self.rollouts.after_update()
        pass


    ############ Utility Function For Update ############

    def feed_forward_generator(self, normalized_advs, num_mini_batch = 8):
        batch_size = self.num_processes * self.num_steps
        mini_batch_size =  batch_size // num_mini_batch

        # sampler is an iterator of indices which size is mini_batch_size

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last= True)

        for indices in sampler:
            obs_batch = self.rollouts.obs[:-1].view(-1, *self.obs_shape)[indices]

            acts_batch = self.rollouts.acts.view(-1, self.act_shape)[indices]

            values_batch = self.rollouts.value_preds[:-1].view(-1, 1)[indices]

            rewards_batch = self.rollouts.rewards.view(-1, 1)[indices]

            returns_batch = self.rollouts.returns[:-1].view(-1, 1)[indices]

            masks_batch = self.rollouts.masks[:-1].view(-1, 1)[indices]

            old_action_log_probs_batch = self.rollouts.action_log_probs.view(-1, 1)[indices]

            yield obs_batch, acts_batch, values_batch, values_batch, rewards_batch, returns_batch, masks_batch, old_action_log_probs_batch






        


        



    def get_critic_value(self, obs):
        critic_value, _ = self.actor_critic_model(obs)
        return critic_value



    ############ Explore or Exploit Part ###############
    # Use in Main Training Loop to Interacte with envs
    # input : obs = (1, num_processes, *obs_shape)
    ####################################################

    def get_action(self, obs, deterministic = False):
        with torch.no_grad():
            values, logits = self.actor_critic_model(obs)
            # obs = (num_processes, *obs_shape)
        
        dist = FixedCategorical(logits = logits)
        
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample() # (num_processes, 1)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return values, action, action_log_probs
        

    


