import torch
import os, glob, json
import numpy as np

from tqdm import tqdm

from datetime import timedelta

from tensorboardX import SummaryWriter
from utils.nqbit_parameters import get_args  

from energy_env.NqubitEnv import NqubitEnv, NqubitEnvContinuous # Env

from agents.SACAgent import SACAgent # Agent


if __name__ == '__main__':

    ########################### args & json & log_dir & writer ###################################
    args = get_args()
    sac_dict = vars(args)

    # log dir & summary writer
    current_dir = './results'
    train_log_dir = '/test_version' + str(args.nbit) + '/sac'
    exp_name = '/seed{0}'.format(args.seed)
    log_dir = current_dir + train_log_dir + exp_name 

    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, 'events.out.tfevents*'))\
            + glob.glob(os.path.join(log_dir, '*.dump')) \
            + glob.glob(os.path.join(log_dir, '*.json'))
        for f in files:
            os.remove(f)
    
    writer = SummaryWriter(log_dir)
    
    with open(os.path.join(log_dir, 'params.json'), 'w') as f:
        f.write(json.dumps(sac_dict, ensure_ascii=False, indent=4, separators=(',',':')))

    ############################## Device & Env & RNG & Agent #####################
    # Device
    device = torch.device("cuda:{}".format(args.GPU))

    # Env
    env = NqubitEnvContinuous(args.nbit, args.episode_length, args.measure_every_n_steps, args.reward_scale)

    # RNG
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.action_space.seed(args.seed)

    # Agent
    agent = SACAgent(args, env, log_dir, device)


    #############################   Main Training Loop ############################
    totalstep = 0
    best_b = None
    best_threshold = -2.0

    # Training Loop
    for episode in tqdm(range(args.num_episodes)): # int(1e6)
        obs = env.reset()  # (9, )


        for step in tqdm(range(args.episode_length)): # (0, 1, 2)
            totalstep += 1
            
            # This if-else is used to increase initial exploration 
            if totalstep > args.random_steps:   # 900
                # required extra exploration strategy
                action = agent.get_action(obs, deterministic = False)
            else:
                action = env.action_space.sample()

            # Excute
            prev_obs = obs
            obs, reward, done, info = env.step(action)
            
            

            # store ( sometimes is wrote into agent.update )
            agent.buffer.store(prev_obs, action, reward, obs, done)

            # when to update & how often we update
            if (totalstep > args.learn_start_steps) and (totalstep % args.update_freq_steps):
                    #value_loss, policy_loss, log_prob_mag, q_value_mag, alpha = agent.update(args.update_freq_per_step, totalstep)
                    value_loss, policy_loss, log_prob_mag, q_value_mag = agent.update(args.update_freq_per_step, totalstep)
                    
                    writer.add_scalar('value_loss', value_loss, totalstep)
                    writer.add_scalar('policy_loss', policy_loss, totalstep)
                    writer.add_scalar('log_prob', log_prob_mag, totalstep)
                    writer.add_scalar('q_value_prob', q_value_mag, totalstep)
                    # logwriter.add_scalar('alpha', alpha, totalstep)
            
            
            # log_state & action
            #if totalstep % args.log_state_action_steps == 0:
                
            #    writer.add_scalars('state_value', {'s0':obs[-6], 's1':obs[-5], 's2':obs[-4], 's3':obs[-3], 's4':obs[-2], 's5':obs[-1]}, totalstep)
            #    writer.add_scalars('log_action', {'a0':action[0], 'a1':action[1], 'a2':action[2], 'a3':action[3], 'a4':action[4], 'a5':action[5]}, totalstep)

            # test_agent
            if (totalstep % args.measure_every_n_steps == 0):
                writer.add_scalar('threshold', info['threshold'], totalstep)
                if info['threshold'] > best_threshold:
                    best_threshold = info['threshold']
                    best_b = info['solution']
                #writer.add_scalar('reward', info['reward'], totalstep)
                #writer.add_scalar('extra_reward', info['extra_reward'], totalstep)
            
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
                #writer.add_scalar('test_episode_reward', avg_reward, totalstep)
            

      
        
    
        measure_state = info['solution']
        episode_reward = info['threshold']
        writer.add_scalar('episode_reward',info['threshold'], episode)
        writer.add_scalars('soluiton', {'s0':measure_state[0], 's1':measure_state[1], 's2':measure_state[2], 's3':measure_state[3],'s4':measure_state[4],'s5':measure_state[5]}, episode)

        #writer.add_scalar('episode_reward', np.sum(np.array(episode_reward)), episode)

    torch.save(agent.model.state_dict(), os.path.join(log_dir, 'sac_model.dump'))
    with open(os.path.join(log_dir, 'solution.text'), 'w') as f:
        f.write('best_threshold:{0}, bset_solution:{1}'.format(best_threshold, best_b))

    env.close()
    
    writer.close()

                
            
 # def main(args, log_dir):
    #pass

# if __name__ == '__main__':
#   for arg in args:
#      log_dir = functionOfArg(arg)
#      main(log_dir, arg)
# how to evaluate the hyperparameters properly



            


            






