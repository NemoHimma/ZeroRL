import torch
import os, glob, json
import numpy as np
from tqdm import tqdm

from tensorboardX import SummaryWriter
from utils.nqbit_parameters import get_td3_args
from energy_env.EnvSetting import OneHotEnv  # Env
from agents.TD3Agent import TD3Agent # Agent

if __name__ == '__main__':
    args = get_td3_args()
    td3_dict = vars(args)

    train_log_dir = './results'
    algo_name = '/latest_version{0}'.format(args.nbit) + '/td3'
    exp_name = '/seed{0}'.format(args.seed)
    log_dir = train_log_dir + algo_name + exp_name

    try:
        os.makedirs(log_dir)
    except OSError:
        pass

    writer = SummaryWriter(log_dir)

    with open(os.path.join(log_dir, 'params.json'), 'w') as f:
        f.write(json.dumps(td3_dict, ensure_ascii=False, indent=4, separators=(',',':')))
    
    device = torch.device("cuda:{}".format(args.GPU))

    # Env
    env = OneHotEnv(args.nbit, args.episode_length, args.measure_every_n_steps, args.reward_scale)
    act_limit = env.action_space.high[0]

    # rng
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.action_space.seed(args.seed)

    # Agent
    agent = TD3Agent(args, env, log_dir, device)

    # Predefined Variable
    totalstep = 0
    best_b = None
    best_threshold = -2.0

    # Training Loop
    for episode in tqdm(range(args.num_episodes)):
        obs = env.reset()
        episode_reward = []
        
        action_noise = (1.0 - episode/args.num_episodes) * args.action_noise
        target_noise = (1.0 - episode/args.num_episodes) * args.target_noise
    # Explore or Exploit
        for step in tqdm(range(args.episode_length)): # 1000
            totalstep += 1

            if totalstep > args.random_steps:  #3000 steps
                action = agent.get_action(obs, action_noise) # log_the_action if you want to
            else:
                action = env.action_space.sample()

        # Execute
            prev_obs = obs
            obs, reward, done, info = env.step(action)
        
   
        #if episode_len == args.max_episode_len:
        #    done = False

        # update part
            

            agent.buffer.store(prev_obs, action, reward, obs, done)

            if (totalstep > args.learn_start_steps) and (totalstep % args.update_freq_steps):
        
                value_loss, policy_loss = agent.update(args.update_freq_per_step, target_noise, totalstep)
                # log info
                writer.add_scalar('value_loss', value_loss, totalstep)
                writer.add_scalar('policy_loss', policy_loss, totalstep)

            if (totalstep % args.measure_every_n_steps ==0):
                writer.add_scalar('step_threshold', info['threshold'], totalstep)
                if info['threshold'] > best_threshold:
                    best_threshold = info['threshold']
                    best_b = info['solution']


        # test_part
        '''
            if (reward >= -1.0):
                tmp = obs
                
                if os.path.isfile(os.path.join(log_dir, 'td3_model.dump')):
                    test_agent = TD3Agent(args, env)
                    test_agent.model.load_state_dict(torch.load(os.path.join(log_dir, 'td3_model.dump')))
                    
                    for _ in range(50):
                        obs = torch.from_numpy(obs).to(args.device)
                        with torch.no_grad():
                            a = test_agent.model.actor(obs)
                        a = a.cpu().numpy()
                        a = np.clip(a, -act_limit, act_limit)
                        obs, reward , done , _ = env.step(a)
                    
                        if (reward < -1.0):
                            obs = tmp
                            break

                    print("reach the goal")
                    print("final_states are {0}, {1}".format(prev_obs, obs))
                else:
                    print('still learning')
        '''
        measure_state = info['solution']
        episode_reward = info['threshold']
        writer.add_scalar('episode_reward',episode_reward, episode)
        writer.add_scalars('soluiton', {'s0':measure_state[0], 's1':measure_state[1], 's2':measure_state[2], 's3':measure_state[3],'s4':measure_state[4],'s5':measure_state[5]}, episode)

    torch.save(agent.model.state_dict(), os.path.join(log_dir, 'td3_model.dump'))
    with open(os.path.join(log_dir, 'solution.text'), 'w') as f:
        f.write('best_threshold:{0}, bset_solution:{1}'.format(best_threshold, best_b))

    env.close()
    writer.close()
        


    

        



    
        













    




    



