import torch
import random

from energy_env.NqubitEnv import NqubitEnvDiscrete, NqubitEnv
from utils.nqbit_parameters import get_args, get_dqn_args
from tensorboardX import SummaryWriter
from agents.DQNAgent import DQNAgent # Agent


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
    env = NqubitEnvDiscrete(args.nbit, initial_state) # env.get_easy_T() remained to do
    agent = DQNAgent(args, env, log_dir, device)
    writer = SummaryWriter(log_dir)
    
    Temp = args.Temp
    totalstep = 0
    epsilon = 1.0

    for episode in range(args.num_episodes):
        Temp = Temp * 10.0 ** (-0.1)
        obs = env.reset(initial_state)
        for step in range (args.episode_length):
            
            # choose large stepsize action number
            action = agent.get_action(obs, epsilon)

            # execute large stepsize number if it satisfies the strong constraint
            next_obs, reward, done, info = env.step(action, args.action_delta)
            
            # judge the large action stepsize effect
            # if ep = 0 : large stepsize is useless

            ep, action_delta = agent.prob(obs, next_obs, action)

            accept_probability = 1 if (ep > 0) else np.exp(ep/Temp)
            u = random.random()
            
            if u <= accept_probability: # take a small stepsize 
        
                next_obs, reward, done, info = env.step(action, action_delta)
            else: # No operation, the transition will be (obs, 0, reward, obs)
                action = 0 
                

            # record
            writer.add_scalar('threshold_rew', reward, totalstep)

            agent.buffer.push((obs, action, reward, next_obs))

            if totalstep > args.learn_start_steps:
                loss = agent.update(totalstep)
                writer.add_scalar('loss', loss, total_step)
                epsilon = agent.epsilon_by_step(totalstep)

            obs = next_obs
            totalstep += 1

            # Test_DQN_Agent
            if (reward >= -1.0):
                test_epsilon = 0.0
                test_obs = obs
                T = env.get_easy_T(args.nbits)

                for step in range(args.test_step):
                    test_action = agent.get_action(test_obs, test_epsilon)

                    # execute large stepsize number
                    test_next_obs, reward, done, info = env.step(test_action, args.action_delta)
                    
                    # judge the large action stepsize effect
                    ep, action_delta = agent.prob(test_obs, test_next_obs, test_action)

                    accept_probability = 1 if (ep > 0) else np.exp(ep/T)
                    u = random.random()

                    if u <= accept_probability: # take a small stepsize 
                    
                        test_next_obs, reward, done, info = env.step(test_action, action_delta)
                    else:
                        action = 0 

                    test_obs = test_next_obs

                if (reward >= -1.0):
                    return reward, test_obs
    
 


if __name__ == '__main__':
    sac_args = get_args()
    dqn_args = get_dqn_args()

    sac_log_dir = './results/sac_energy_new/nbit-9'  # choose one of the sac_model.dump
    sac_device = torch.device("cuda:1")
    dqn_log_dir = ''  # where to save the model
    

    #initial_state = SAC_initial_state(sac_args, log_dir, device)
    initial_state = [-1.0713e-3,-1.1502e-3,-4.3445e-3, -5.1295e-3,-7.6635e-4, 6.2604e-3]
    print(initial_state)
    #for i in range(len(initial_state)): #8
    #    dqn_device = torch.device("cuda:{}").format(i % 2)
        ### parallel run DQN_Exploration (required passing initial_state)
    #    reward, test_obs = DQN_Exploration(dqn_args, dqn_device, initial_state[i])


    #  Parallel_run DQN_Exploration
    # record the exploration
    
