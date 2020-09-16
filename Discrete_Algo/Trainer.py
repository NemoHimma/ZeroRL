import os, glob
import torch

from timeit import default_timer as timer
from datetime import timedelta

from utils.hyperparameters import Config

from envs.AtariEnv import PrepareAtariEnv

from agents.DQNAgent import DQNAgent

from tqdm import tqdm

from utils.plot import plot_all_data

if __name__ == '__main__':
    start = timer()
    exp_name = 'DQN4'
    log_dir = './'+ exp_name + '/' 
    env_id = 'PongNoFrameskip-v4'
    
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))\
            + glob.glob(os.path.join(log_dir, '*td.csv')) \
            + glob.glob(os.path.join(log_dir, '*sig_param_mag.csv'))  \
            + glob.glob(os.path.join(log_dir, '*action_log.csv'))

        for f in files:
            os.remove(f)
    
    # Config
    config = Config()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Env
    env = PrepareAtariEnv(env_id, log_dir)

    # Agent
    agent = DQNAgent(config, env, log_dir, static_policy=False)

    # Begin Interaction & Learning
    
    episode_reward = 0
    observation = env.reset()

    for frame_idx in tqdm(range(1, config.MAX_FRAMES+1)):
        # Prepare to explore
        eps = agent.epsilon_by_frame(frame_idx)

        # Explore or Exploit
        action = agent.get_action(eps, observation)
        agent.save_action(action)

        # Execute
        prev_observation = observation
        observation, reward, done, info = env.step(action)

        if done:
            observation = None

        # Learn
        agent.update(prev_observation, action, reward, observation)
        episode_reward += reward

        # Episode End

        if done:
            agent.finish_nstep()
            agent.save_reward(episode_reward)

            observation = env.reset()
            episode_reward = 0

        # Log Info 
        if frame_idx % 10000 == 0:
            agent.save_weight()
            try:
                plot_all_data(log_dir, env_id, exp_name, config.MAX_FRAMES, bin_size=(10, 100, 100, 1), save_filename = exp_name + '.png', smooth=1, time=timedelta(seconds=int(timer()-start)), ipynb=False)

            except IOError:
                pass



    






