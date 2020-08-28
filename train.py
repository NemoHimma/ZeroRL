import numpy as np
import math, glob, os

from timeit import default_timer as timer
from datetime import timedelta
from tqdm import tqdm 
# env & Monitor & Wrapper
import gym 
from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from wrappers import WrapPyTorch

# Agent
from agents import DQNAgent

# Config
from hyperparameters import Config 

# Plot data
from plot import plot_all_data 

if __name__ == "__main__":
    start = timer()
    log_dir = './experiments/'

    # make_dirs or remove 
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv')) \
            + glob.glob(os.path.join(log_dir, '*td.csv')) \
            + glob.glob(os.path.join(log_dir, '*sig_param_mag.csv')) \
            + glob.glob(os.path.join(log_dir, '*action_log.csv'))
        for f in files:
            os.remove(f)

    # config
    config = Config()

    # env
    env_id = "PongNoFrameskip-v4"
    env = make_atari(env_id)
    env = bench.Monitor(env, os.path.join(log_dir, env_id))
    env = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=True)
    env = WrapPyTorch(env)

    # agent
    agent = DQNAgent(config = config, env = env, log_dir = log_dir)

    ## Interaction & Learning
    episode_reward = 0
    observation = env.reset()

    for frame_idx in tqdm(range(1, config.MAX_FRAMES+1)):
        # epsilon-greedy by frames
        eps = config.epsilon_by_frame(frame_idx)

        # select action based on eps&observation then log it
        action = agent.get_action(observation, eps)
        agent.save_action(action, frame_idx)

        # excution step
        prev_observation = observation
        observation , reward , done , _ = env.step(action)
        if done:
            observation = None
        
        # main-step
        agent.update(prev_observation, action , reward , observation, frame_idx) 
        episode_reward += reward

        # episode end
        if done:
            agent.finish_nstep()
            observation = env.reset()
            agent.save_reward(episode_reward)
            episode_reward = 0

        # save the model
        if frame_idx % 10000 == 0:
            agent.save_weight()
            try:
                print('frame %s. time: %s' % (frame_idx, timedelta(seconds=int(timer()-start))))
                plot_all_data(log_dir, env_id, 'DQN', config.MAX_FRAMES, bin_size=(10, 100, 100, 1), smooth=1, time=timedelta(seconds=int(timer()-start)), ipynb=False)
            except IOError:
                pass

    agent.save_weight()
    env.close()