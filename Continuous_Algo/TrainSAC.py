import os
import glob
import numpy as np


import torch
# args
from utils.SACArgument import get_args
# Log
from tensorboardX import SummaryWriter
from tqdm import tqdm
# Env
import gym
# Buffer
from buffer.DDPGBuffer import DDPGReplayBuffer
# Agent
from agents.SACAgent import SACAgent

'''
Running Statics Mean/Std Remember to add in PPO
'''

if __name__ == '__main__':
    args = get_args()

    # specific args
    args.seed = 123456
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # log_dir
    prefix = './results/'
    log_dir = os.path.join(prefix , args.env_id + '/' + args.algo)
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.pami12')) + glob.glob(os.path.join(log_dir, '*.dump'))
        for f in files:
            os.remove(f)

    # tensorboardX
    writer = SummaryWriter(log_dir)

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Env & Extract Info from envs (obs_shape, act_shape)
    env = gym.make(args.env_id)
    test_env = gym.make(args.env_id)

    # Assumed to Use Mujoco (Which can be writtened into Function For Different Envs
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    # Buffer (input nd.array output nd.array)
   
    buffer = DDPGReplayBuffer(obs_dim, act_dim, args.buffer_size)

    # Agent
    agent = SACAgent(obs_dim, act_dim, act_limit, args, device, writer)

    # Training Loop
    obs = env.reset()
    episode_reward = 0. 
    episode_length = 0
    num_episodes = 0
    
    for step in tqdm(range(args.total_env_steps)):

        # Exploration or Exploitation
        if (step < args.random_steps):
            action = env.action_space.sample()
        else:
            action = agent.get_action(obs)

        # Excute
        next_obs, reward, done, info = env.step(action)
        episode_reward += reward
        episode_length += 1

        # TimeLimit or Real Done 
        done = False if episode_length == env._max_episode_steps else done

        # store
        buffer.store(obs, action, reward, next_obs, done)

        obs = next_obs

        # if Real Done
        if done:
            num_episodes += 1
            writer.add_scalar('Train_Episode_Reward', episode_reward, num_episodes)
            writer.add_scalar('Train_Episode_Length', episode_length, num_episodes)
            obs = env.reset()
            episode_reward = 0
            episode_length = 0

        # update 
        if step > args.learn_start_steps:
            if (step + 1) % args.update_every_steps == 0:
                policy_loss, value_loss = agent.update(buffer, step)
                writer.add_scalar('policy_loss', policy_loss, step)
                writer.add_scalar('value_loss', value_loss, step)
            
            

        # log state & action
        if (step + 1) % 100 == 0:
             writer.add_scalars('state_value', {'s' + str(i) : obs[i] for i in range(obs_dim)}, step)
             writer.add_scalars('action_value', {'a' + str(j): action[j] for j in range(act_dim)}, step)

        # test_agent & save model
        if (step + 1) % (args.total_env_steps/10) == 0:
            avg_reward = 0.
            test_episodes = 5
            for _ in range(test_episodes):
                obs, done, ep_rew, ep_len = test_env.reset(), False, 0.0, 0
                while not (done or (ep_len == test_env._max_episode_steps)):
                    action = agent.get_action(obs, deterministic = True)
                    next_obs, reward, done, info = env.step(action)
                    ep_rew += reward
                    ep_len += 1
                    obs = next_obs
                
                avg_reward += ep_rew
            
            avg_reward /= test_episodes
            # save_model
            torch.save(agent.model.state_dict(), os.path.join(log_dir, 'sac_model.dump'))
            writer.add_scalar('Test_Episode_Reward', avg_reward, step)

    
    env.close()










        





        

        




    

    

    
