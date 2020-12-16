import torch
import random

from energy_env.NqubitEnv import NqubitEnvDiscrete, NqubitEnv
from utils.nqbit_parameters import get_args, get_dqn_args




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
    return initial_state



def DQN_Exploration(args, log_dir, device, initial_state):
    env = NqubitEnvDiscrete(args.nbits, initial_state)
    agent = DQNAgent(args, env, log_dir, device)

    Temp = 10

    for episode in range(80):
        Temp = Temp * 10 ** (-0.1)
        obs, done, ep_rew = env.reset(), False, 0.0 
        for step in range (1000):
            totalstep = episode * 1000 + step + 1

            action = agent.get_action(obs)

            next_obs, reward, done, info = env.step(action)

            ep, action_delta = agent.prob(obs, next_obs, action)

            accept_probability = 1 if (np.exp(ep/Temp)> 1) else np.exp(ep/Temp)
            u = random.random()
            if u <= accept_probability:
                






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
    
