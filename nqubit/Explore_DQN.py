import torch
import random

from energy_env.NqubitEnv import NqubitEnvDiscrete, NqubitEnv
from utils.nqbit_parameters import get_args, get_dqn_args
from tensorboardX import SummaryWriter




def SAC_initial_state(args, log_dir, device):
    env = NqubitEnv(args.max_episode_steps, args.nbits, args.T)

    agent = SACAgent(args, env, log_dir, device)

    torch.load(os.path.join(log_dir, 'sac_model.dump'), agent.model.state_dict())

    initial_state = []
    for episode in range(args.n_initial_points):
        obs, done, ep_rew = env.reset(), False, 0.0
        for i in range(args.max_episode_steps):
            action = agent.get_action(obs, deterministic = True)
            next_obs, reward, done, info = env.step(action)
            ep_rew += reward
            obs = next_obs
        
        initial_state.append(info['solution'])
    
    env.close()
    print(initial_state)
    return initial_state



def DQN_Exploration(args, log_dir, device, initial_state):
    env = NqubitEnvDiscrete(args.nbits, initial_state) # env.get_easy_T() remained to do
    agent = DQNAgent(args, env, log_dir, device)
    writer = SummaryWriter(log_dir)

    Temp = 10

    for episode in range(80):
        Temp = Temp * 10 ** (-0.1)
        obs, done, ep_rew = env.reset(), False, 0.0 
        for step in range (1000):
            totalstep = episode * 1000 + step + 1
            
            epsilon = agent.exploration_strategy(totalstep) # from 1 to 0
            # choose large stepsize action number
            action = agent.get_action(obs, epsilon)

            # execute large stepsize number
            next_obs, reward, done, info = env.step(action, action_delta=0.001)
            
            # judge the large action stepsize effect
            ep, action_delta = agent.prob(obs, next_obs, action)

            accept_probability = 1 if (ep > 0) else np.exp(ep/Temp)
            u = random.random()

            if u <= accept_probability: # take a small stepsize 
                
                action = agent.get_action(obs)

                next_obs, reward, done, info = env.step(action, action_delta)
            else:
                action = 0 

                next_obs, reward, done, info = env.step(action, action_delta)
            
            writer.add_scalar('threshold_rew', reward, totalstep)
            agent.buffer.store(obs, action, reward, next_obs)

            if totalstep > args.learn_start_steps:
                agent.update()

            obs = next_obs

            if (reward >= -1.0):
                test_epsilon = 0.0
                T = env.get_easy_T(args.nbits)

                for step in range(200):
                    action = agent.get_action(obs, test_epsilon)

                    # execute large stepsize number
                    next_obs, reward, done, info = env.step(action, action_delta=0.001)
                    
                    # judge the large action stepsize effect
                    ep, action_delta = agent.prob(obs, next_obs, action)

                    accept_probability = 1 if (ep > 0) else np.exp(ep/T)
                    u = random.random()

                    if u <= accept_probability: # take a small stepsize 
                        
                        action = agent.get_action(obs, test_epsilon)

                        next_obs, reward, done, info = env.step(action, action_delta)
                    else:
                        action = 0 

                        next_obs, reward, done, info = env.step(action, action_delta)









    pass

if __name__ == '__main__':
    sac_args = get_args()
    dqn_args = get_dqn_args()

    sac_log_dir = ''  # choose one of the sac_model.dump
    sac_device = torch.device("cuda:1")
    dqn_log_dir = ''

    initial_state = SAC_initial_state(sac_args, log_dir, device)
    for i in range(len(initial_state)): #8
        dqn_device = torch.device("cuda:{}").format(i % 2)
        ### parallel run DQN_Exploration (required passing initial_state)
        DQN_Exploration(dqn_args, dqn_device, initial_state[i])


    #  Parallel_run DQN_Exploration
    # record the exploration
    
