# Utility Package
import os, glob
from tqdm import tqdm
from timeit import default_timer as timer
import numpy as np

# Model, Env, Buffer, Agent
import torch
import gym

from buffer.ddpgBuffer import DDPGReplayBuffer
from agents.DDPGAgent import DDPGAgent
from utils.ddpghyperparameters import Config

from baselines import bench

    start = timer()

# Configuration 
    config = Config()
    algo_name = 'ddpg'
    log_dir = './results/' + algo_name + '/'

    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


# Specific Configuration
    config.device = torch.device("cuda:0")

# Seed rng
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

# Env & Extract Info from env
    env_id = 'HalfCheetah-v2'
    env = gym.make(env_id)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape[0] # Box Type Not Discrete
    env = bench.Monitor(env, os.path.join(log_dir)) # record reward, length, time

# Agent
    agent = DDPGAgent(config, env)

# Predefined Variable
episode_reward = 0
observation = env.reset()
episode_len = 0

total_num_steps = config.epochs * config.per_epoch_steps
# Main Trainning Loop
for timestep in tqdm(range(total_num_steps)):
    #############################
    # Explore or Exploit 
    if timestep > config.start_to_exploit_steps:
        # When Exploit , Add Some Noise
        action = agent.get_action(obseration, config.action_noise)
    else: 
        action = env.action_space.sample()
###################################
    prev_observation = observation

    observation, reward, done , _ = env.step(action)
    episode_reward += reward
    episode_len += 1

    if episode_len == config.max_episode_len:
        done = False

    agent.model.train()

    agent.update(prev_observation, action, reward, observation, done, timestep)

# If done reset 
    if done or (episode_len == config.max_episode_len):
        observation, episode_reward, episode_len = env.reset(), 0 , 0

# Save & log
    if (timestep+1) % config.per_epoch_steps == 0:
        epoch = (t + 1) // config.per_epoch_steps

        if (epoch % config.save_model_freq == 0) or (epoch == config.epochs):
            torch.save(agent.model.parameters(), os.path.join(log_dir,'ddpg_model.pt'))

    


    


