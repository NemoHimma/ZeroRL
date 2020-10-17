import torch
import os, glob
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
from collections import deque


from tensorboardX import SummaryWriter
from utils.nqbit_parameters import Config  
from envs.Nqubits import NqubitEnv  # Env
from agents.DDPGAgent import DDPGAgent # Agent

if __name__ == '__main__':
    config = Config()
    start = timer()

    train_log_dir = './results/'
    algo_name = 'ddpg/'
    log_dir = train_log_dir + algo_name
    writer = SummaryWriter(log_dir)
    


    try:
        os.makedirs(log_dir)
    except OSError:
        pass

    # Specific Configuration
    config.device = torch.device("cuda:0")

    # rng
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Env
    env = NqubitEnv.NqubitEnv2()
    act_limit = env.action_space.high[0]

    # Agent
    agent = DDPGAgent(config, env)

    # Predefined Variable
    

    # Training Loop
    for episode in tqdm(range(config.num_episodes)):
        obs = env.reset()
        episode_reward = []
        running_reward = deque(maxlen = 100)
    # Explore or Exploit
        for step in range(config.max_episode_steps):

            if step > config.start_to_exploit_steps:  #10000 steps
                action = agent.get_action(obs, config.action_noise) # log_the_action if you want to
            else:
                action = env.action_space.sample()

        # Excute
            prev_obs = obs
            obs, reward, done,  _ = env.step(action)
        
            running_reward.append(reward)
            episode_reward.append(reward)

        #if episode_len == config.max_episode_len:
        #    done = False

        # update part
            timestep = episode * config.max_episode_steps + step + 1
            writer.add_scalar('Step_reward', reward, timestep)

            agent.buffer.store(prev_obs, action, reward, obs)

            if timestep > config.learn_start_steps:
                if (step > 50) and (step % 2 == 0):  # Choose When to Update
                    value_loss, policy_loss = agent.update()
                # log info
                    writer.add_scalar('value_loss', value_loss, timestep)
                    writer.add_scalar('policy_loss', policy_loss, timestep)

        # test_part
            if (reward >= -1.0):
                tmp = obs
                
                if os.path.isfile(os.path.join(log_dir, 'ddpg_model.dump')):
                    test_agent = DDPGAgent(config, env)
                    test_agent.model.load_state_dict(torch.load(os.path.join(log_dir, 'ddpg_model.dump')))
                    
                    for _ in range(50):
                        obs = torch.from_numpy(obs).to(config.device)
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

        
        # log part
            if (episode + 1) % config.print_freq == 0:
                end = timer()
                print("Train Time:{:.1f}, epoch:{}, mean/median reward {:.4f}/{:.4f}, min/max reward {:.4f}/{:.4f}".format((end - start), episode + 1, 
                np.mean(running_reward),
                np.median(running_reward),
                np.min(running_reward),
                np.max(running_reward)))

                if ((episode + 1) % config.save_model_freq == 0) or ((episode+1) == config.num_episodes):
                    torch.save(agent.model.state_dict(), os.path.join(log_dir, 'ddpg_model.dump'))
        writer.add_scalar('episode_reward', np.mean(episode_reward), episode)

    

        



    
        













    




    



