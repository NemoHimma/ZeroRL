import torch
import random
import os
import numpy as np
from tqdm import tqdm
import json

from energy_env.NqubitEnv import NqubitEnvDiscrete, NqubitEnv
from utils.nqbit_parameters import get_args, get_dqn_args
from tensorboardX import SummaryWriter
from agents.DQNAgent import DQNAgent # DQNAgent
from agents.SACAgent import SACAgent # SACAgent


def SAC_initial_state(args, log_dir, device):
    env = NqubitEnv(args.max_episode_steps, args.nbit, args.T)

    agent = SACAgent(args, env, log_dir, device)

    torch.load(os.path.join(log_dir, 'sac_model.dump'), agent.model.state_dict())

    
    obs, done = env.reset(), False
    for i in range(args.max_episode_steps):
        action= agent.get_action(obs, deterministic = True)
        next_obs, reward, done, info = env.step(action)
        #ep_rew += reward
        obs = next_obs
    
    initial_state = info['solution']
    corresponding_threshold = info['threshold']
    
    env.close()
    return initial_state, corresponding_threshold



def DQN_Exploration(args, log_dir, device, initial_state):
    env = NqubitEnvDiscrete(args.nbit, initial_state) # env.get_easy_T() remained to do
    agent = DQNAgent(args, env, log_dir, device)
    writer = SummaryWriter(log_dir)
    
    Temp = args.Temp
    totalstep = 0
    epsilon = 1.0
    obs = env.reset()
    print('initial_reward{0}'.format(env.get_current_threshold(obs)))
    

    for episode in tqdm(range(args.num_episodes)):
        Temp = Temp * 10.0 ** (-0.1)
        obs = env.reset()
        
        for step in tqdm(range (args.episode_length)):
            
            # choose large stepsize action number
            action = agent.get_action(obs, epsilon)
            # aciton <class 'int'>
            
            # execute large stepsize number if it satisfies the strong constraint
            next_obs, reward, done, info = env.step(obs, action, args.action_delta)
            #agent.buffer.push((obs, action, reward, next_obs))
            
            # judge the large action stepsize effect
            # if ep = 0 : large stepsize is useless

            ep, action_delta = agent.prob(obs, next_obs, action)

            accept_probability = 1 if (ep > 0) else np.exp(ep/Temp)
            u = random.random()
            
            if u <= accept_probability: # take a small stepsize 
                #agent.buffer.push((obs, action, reward, next_obs))

                next_obs, reward, done, info = env.step(obs, action, action_delta)
            else: # No operation, the transition will be (obs, 0, reward, obs)
                action = 0 
                next_obs, reward, done, info = env.step(obs, action, action_delta)
                

            # record
            writer.add_scalar('threshold_rew', reward, totalstep)

            agent.buffer.push((obs, action, reward, next_obs))

            if (totalstep > args.learn_start_steps) and (totalstep % args.update_freq==0):
                loss = agent.update()
                writer.add_scalar('loss', loss, totalstep)
                epsilon = agent.epsilon_by_step(totalstep)
                if epsilon < args.epsilon_min:
                    epsilon = args.epsilon_min

            obs = next_obs
            totalstep += 1
            if (reward >= -1.0):
                return reward, obs

            # Test_DQN_Agent
            if (totalstep % args.test_freq == 0):
                test_epsilon = 0.0
                test_obs = env.reset()
                #T = env.get_easy_T(args.nbits)
                reward_recorder = -2.0
                obs_recorder = test_obs
                

                for step in range(args.test_step):
                    test_action = agent.get_action(test_obs, test_epsilon)

                    # execute large stepsize number
                    test_next_obs, reward, done, info = env.step(test_obs, test_action, args.action_delta)
                    
                    # judge the large action stepsize effect
                    ep, action_delta = agent.prob(test_obs, test_next_obs, test_action)

                    accept_probability = 1 if (ep > 0) else np.exp(ep/Temp)
                    u = random.random()

                    if u <= accept_probability: # take a small stepsize 
                    
                        test_next_obs, reward, done, info = env.step(test_obs, test_action, action_delta)
                    else:
                        action = 0
                        test_next_obs = test_obs
                        reward = env.get_current_threshold(test_obs)
                        
    
                    if reward > reward_recorder:
                        reward_recorder = reward
                        obs_recorder = test_next_obs
                    if (reward >= -1.0):
                        return reward, test_obs
                    
                    agent.buffer.push((test_obs, action, reward, test_next_obs))
                    test_obs = test_next_obs
                
                writer.add_scalar('test_max_reward', reward_recorder, totalstep)
                writer.add_scalars('solution', {'s0':obs_recorder[0],'s1':obs_recorder[1],'s2':obs_recorder[2],'s3':obs_recorder[3],'s4':obs_recorder[4],'s5':obs_recorder[5]}, totalstep)


                
    
 


if __name__ == '__main__':
    '''
    # arg, log_dir, device
    sac_args = get_args()
    sac_log_dir = './results/sac_energy_new/nbit-9/T-9.200seed-1/'  # choose one of the sac_model.dump
    sac_device = torch.device("cuda:1")
    
    # arg specific configuration
    sac_args.T = 9.200
    sac_args.nbit = 9
    sac_args.n_initial_points = 10
    initial_state_n, corresponding_threshold_n = [], []
    #initial_state = [-1.0713e-3,-1.1502e-3,-4.3445e-3, -5.1295e-3,-7.6635e-4, 6.2604e-3]
    for i in range(sac_args.n_initial_points):
        initial_state, corresponding_threshold = SAC_initial_state(sac_args, sac_log_dir, sac_device)
        initial_state_n.append(initial_state)
        corresponding_threshold_n.append(corresponding_threshold)

    max_index = np.array(corresponding_threshold_n).argmax()
    initial_state = np.array(initial_state_n[max_index])

    print(initial_state_n)
    print(corresponding_threshold_n)
    print(max_index)
    print(initial_state)
    '''

    # dqn_exploration part
    dqn_args = get_dqn_args()
    dqn_device = torch.device("cuda:1")
    
    # dqn log_dir

    current_dir = './results/'
    train_log_dir = 'dqn_exploration/'
    exp_name = 'nbit-9T-9.200-original-setting/'
    dqn_log_dir = current_dir + train_log_dir + exp_name

    try:
        os.makedirs(dqn_log_dir)
    except OSError:
        pass

    initial_state = [-0.01515305, -0.0022998, -0.00602785, 0.00275275, 0.01112909, -0.00420499]
    
    # specific configuration
    dqn_args.nbit = 9
    dqn_dict = vars(dqn_args)
    # Env T has been set to 9.200
    with open(dqn_log_dir + 'paras.json', 'w') as f:
        f.write(json.dumps(dqn_dict, ensure_ascii=False, indent=4, separators=(',', ':')))

    reward, test_obs = DQN_Exploration(dqn_args, dqn_log_dir, dqn_device, initial_state)
    print(test_obs, reward)
    
    #for i in range(len(initial_state)): #8
    #    dqn_device = torch.device("cuda:{}").format(i % 2)
        ### parallel run DQN_Exploration (required passing initial_state)
    #    reward, test_obs = DQN_Exploration(dqn_args, dqn_device, initial_state[i])


    #  Parallel_run DQN_Exploration
    # # record the exploration
    
