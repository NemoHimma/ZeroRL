import torch
import os, glob
import numpy as np

from tqdm import tqdm
from timeit import default_timer as timer
from datetime import timedelta

from collections import deque
from torch.autograd import Variable


from tensorboardX import SummaryWriter
from utils.nqbit_parameters import Config  
from envs.Nqubits import NqubitEnv  # Env
from agents.SACAgent import SACAgent # Agent

if __name__ == '__main__':
    config = Config()
    start = timer()

    train_log_dir = './results/'
    algo_name = 'sac_exp_reward/'
    log_dir = train_log_dir + algo_name
    writer = SummaryWriter(log_dir)
    try:
        os.makedirs(log_dir)
    except OSError:
        pass

    # Specific Configuration for SAC
    config.alpha = 0.02
    config.device = torch.device("cuda:1")

    # RNG
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Env
    env, test_env = NqubitEnv.NqubitEnv2(), NqubitEnv.NqubitEnv2()
    act_limit = env.action_space.high[0]

    # Agent
    agent = SACAgent(config, env)


    # Training Loop
    for episode in tqdm(range(config.num_episodes)):
        obs = env.reset()
        episode_reward = []
        running_reward = deque(maxlen = 20)

        for step in range(config.max_episode_steps):
            timestep = episode * config.max_episode_steps + step + 1
            
            # This if-else is used to increase initial exploration 
            if timestep > config.start_to_exploit_steps:
                action = agent.get_action(obs, deterministic = False)
            else:
                action = env.action_space.sample()

            # Excute
            prev_obs = obs
            obs, reward, done, _ = env.step(action)
            
            # log info
            writer.add_scalar('Step_Reward', reward, timestep)
            episode_reward.append(reward)
            running_reward.append(reward)

            # store ( sometimes is wrote into agent.update )
            agent.buffer.store(prev_obs, action, reward, obs, done)

            # when to update & how often we update
            if timestep > config.learn_start_steps:
                if ((step + 1) >= 50) and ((step + 1) % 50 == 0):
                    value_loss, policy_loss = agent.update(50, timestep)
                    
                    writer.add_scalar('value_loss', value_loss, timestep)
                    writer.add_scalar('policy_loss', policy_loss, timestep)
            
            # log_state & action
            if (step + 1) % 50 == 0:
                writer.add_scalars('state_value', {'s0':obs[0], 's1':obs[1], 's2':obs[2], 's3':obs[3], 's4':obs[4], 's5':obs[5]}, timestep)
                writer.add_scalars('log_action', {'a0':action[0], 'a1':action[1], 'a2':action[2], 'a3':action[3], 'a4':action[4], 'a5':action[5]}, timestep)

            # test_agent
            if (reward >= -2.30):
                tmp = obs
                count = 0
                for _ in range(200):
                    count += 1
                    action = agent.get_action(obs, deterministic = True)
                    obs, reward, done, _ = env.step(action)

                    if (reward < -1.0):
                        obs = tmp
                        print("still learning")
                        break

                if count == 200:
                    print("reach the goal")
                    print("final_state is {0}".format(obs))
                    writer.add_scalars('success_state_value', {'s0':obs[0],'s1':obs[1], 's2':obs[2], 's3':obs[3], 's4':obs[4], 's5':obs[5]}, episode)
            
 

        if (episode + 1) % config.print_freq == 0:
            end = timer()
            print("Train Time:{}, episode:{}, mean/median reward {:.4f}/{:.4f}, min/max reward {:.4f}/{:.4f}".format(timedelta(seconds = int(end - start)), episode + 1, 
                np.mean(running_reward),
                np.median(running_reward),
                np.min(running_reward),
                np.max(running_reward)))

        if ((episode + 1) % config.save_model_freq == 0) or ((episode + 1) == config.num_episodes):
            torch.save(agent.model.state_dict(), os.path.join(log_dir, 'sac_model.dump'))

        writer.add_scalar('episode_reward', np.mean(episode_reward), episode)


            


            






