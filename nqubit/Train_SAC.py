import torch
import os, glob
import numpy as np

from tqdm import tqdm
from timeit import default_timer as timer
from datetime import timedelta

from collections import deque
from tensorboardX import SummaryWriter
from utils.nqbit_parameters import get_args  

from energy_env.NqubitEnv import NqubitEnv  # Env

from agents.SACAgent import SACAgent # Agent


if __name__ == '__main__':
    args = get_args()
    start = timer()

    # threshold 
    satisfied_flag = 0
    convergence_buffer = []

    # log dir & summary writer
    current_dir = './results/'
    train_log_dir = '/sac_energy_new/' + 'nbit-' + str(args.nbit)

    #exp_name = 'steps-' + str(args.max_episode_steps) + 'actor_hidden_size-' + str(args. actor_hidden_size) + 'gamma-' + str(args.gamma) + 'batch_size-' + str(args.batch_size) + 'T2.50' 

    exp_name = '/T-' + str(format(args.T, '.3f')) + 'seed-' + str(args.seed)
    log_dir = current_dir + train_log_dir + exp_name
    
    writer = SummaryWriter(log_dir)

    try:
        os.makedirs(log_dir)
    except OSError:
        pass
    

    # Device
    device = torch.device("cuda:{}".format(args.GPU))

    # RNG
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Env
    env, test_env = NqubitEnv(args.max_episode_steps, args.nbit, args.T), NqubitEnv(args.max_episode_steps)
    

    # Agent
    agent = SACAgent(args, env, log_dir, device)


    # Training Loop
    for episode in tqdm(range(args.num_episodes)): # int(1e6)
        obs = env.reset()  # (9, )
        episode_reward = []

        for step in range(args.max_episode_steps): # (0, 1, 2)
            timestep = episode * args.max_episode_steps + step + 1
            
            # This if-else is used to increase initial exploration 
            if timestep > args.start_to_exploit_steps:   # 900
                action = agent.get_action(obs, deterministic = False)
            else:
                action = env.action_space.sample()

            # Excute
            prev_obs = obs
            obs, reward, done, info = env.step(action)
            
            episode_reward.append(reward)
            # log info
                      

            # store ( sometimes is wrote into agent.update )
            agent.buffer.store(prev_obs, action, reward, obs, done)

            # when to update & how often we update
            if timestep > args.learn_start_steps:
                    value_loss, policy_loss, log_prob_mag, q_value_mag = agent.update(args.update_freq_per_step, timestep)
                    
                    writer.add_scalar('value_loss', value_loss, timestep)
                    writer.add_scalar('policy_loss', policy_loss, timestep)
                    writer.add_scalar('log_prob', log_prob_mag, timestep)
                    writer.add_scalar('q_value_prob', q_value_mag, timestep)
            
            
            # log_state & action
            #if timestep % args.log_state_action_steps == 0:
                
            #    writer.add_scalars('state_value', {'s0':obs[-6], 's1':obs[-5], 's2':obs[-4], 's3':obs[-3], 's4':obs[-2], 's5':obs[-1]}, timestep)
            #    writer.add_scalars('log_action', {'a0':action[0], 'a1':action[1], 'a2':action[2], 'a3':action[3], 'a4':action[4], 'a5':action[5]}, timestep)

            # test_agent
            
            if info and (info['threshold'] >= -1.05):
                pass
                '''
                if satisfied_flag == 0:
                    satisfied_flag = episode
                elif ((episode - satisfied_flag)== 1):   
                    convergence_buffer.append(info['threshold'])
                    satisfied_flag == episode
                else:
                    satisfied_flag == episode
                    
                          
                if len(convergence_buffer) == 100:
                    mean = np.mean(np.array(convergence_buffer))
                    if (mean >= -1.005):
                        #torch.save(agent.model.state_dict(), os.path.join(log_dir, 'sac_model.dump'))
                '''
            

                '''

                avg_reward = 0.
                test_episodes = 5
                for _ in range(test_episodes):
                    obs, done, ep_rew  = test_env.reset(), False, 0.0
                    for i in range(args.max_episode_steps): # 3
                        action = agent.get_action(obs, deterministic = True)
                        next_obs, reward, done, info = env.step(action)
                        ep_rew += reward
                        obs = next_obs

                    avg_reward += ep_rew
                
                avg_reward /= test_episodes

                '''
            
                #torch.save(agent.model.state_dict(), os.path.join(log_dir, 'sac_model.dump'))
                #writer.add_scalar('test_episode_reward', avg_reward, timestep)
            

      
        writer.add_scalar('threshold', info['threshold'], episode)
        if episode > (3 * int(args.start_to_exploit_steps/args.max_episode_steps)):
            measure_state = info['solution']
            writer.add_scalars('soluiton', {'s0':measure_state[0], 's1':measure_state[1], 's2':measure_state[2], 's3':measure_state[3],'s4':measure_state[4],'s5':measure_state[5]}, episode)

        writer.add_scalar('episode_reward', np.sum(np.array(episode_reward)), episode)

    env.close()
    test_env.close()
    writer.close()

                
            
 # def main(args, log_dir):
    #pass

# if __name__ == '__main__':
#   for arg in args:
#      log_dir = functionOfArg(arg)
#      main(log_dir, arg)
# how to evaluate the hyperparameters properly



            


            






