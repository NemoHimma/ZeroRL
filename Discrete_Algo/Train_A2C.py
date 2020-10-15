import torch
import os, glob
import numpy as np
from collections import deque
import time
from tqdm import tqdm

from utils.PGhyperparameters import Config
from envs.ikostrov_envs import make_vec_envs
from envs.ikostrov_envs import VecNormalize
from agents.A2CAgent import A2CAgent


def makedirCleanUp(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


if __name__ == '__main__':
    # parameters config
    config = Config()

    # seed rng in CPU & GPU
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    # appoint log_dir & create
    train_log_dir = './results/A2C/'

    makedirCleanUp(train_log_dir)

    # appoint num_thread for parallel envs & device
    torch.set_num_threads(1)
    config.device = torch.device("cuda:0")

    # create envs
    env_id = 'PongNoFrameskip-v4'
    envs = make_vec_envs(env_id, config.seed, config.num_processes, train_log_dir, config.device, False)

    # Agent
    a2cAgent = A2CAgent(config, envs)

    # rollots buffer in Agent
    # rollouts = RolloutBuffer(config.num_steps, config.num_processes, envs.observation_space.shape, envs.action_space)


    # Initialization
    obs = envs.reset()  # all tensor operations / pay attention to this
    a2cAgent.rollouts.obs[0].copy_(obs)  # has stored initial obs
    #rollouts.to_device(config.device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    
    for i in tqdm(range(config.num_updates)):
        # Decay the Learning Rate If Necessary
        if config.USE_DECAY_LR:
            config.LinearDecayLR(a2cAgent.optimizer, i , config.num_updates, config.lr)
        
        for step in range(config.num_steps):
            # sample action
            with torch.no_grad():
                values, action, action_log_probs = a2cAgent.get_action(a2cAgent.rollouts.obs[step]) # (1, num_processes, frame_stack_channels, h, w)

            # excute action in envs
            obs, rewards, done, infos = envs.step(action)
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
            
            # process masks & bad_masks
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])  # TimeLimitMask Wrapper
            
            # store(obs, actions, action_log_probs, values, rewards, masks, bad_masks)
            a2cAgent.rollouts.store(obs, action, rewards, action_log_probs, values, masks, bad_masks)

        # deals with truncated epsiode
        #with torch.no_grad():
        #    next_value = a2cAgent.get_critic_value(rollouts.obs[-1]).detach()

        # compute return
        #rollouts.compute_returns(next_value, config.gamma, config.gae_lambda, config.USE_PROPER_TIME_LIMITS)

        # update
        value_loss, action_loss, dist_entropy = a2cAgent.update()

        # reset rollouts which can also be written into update()
        a2cAgent.rollouts.after_update()


        ################ Save & Log #########################
        '''
        
        Plot Results Remained to be finished

        '''

        if (((i+1) % config.save_model_freq == 0) or ((i+1) % config.num_updates ==0)) :
            
            torch.save([a2cAgent.actor_critic_model, getattr(get_vec_normalize(envs), 'ob_rms', None)], os.path.join(train_log_dir , env_id + ".pt"))

        if ((i+1) % config.episode_rewards_freq ==0) and len(episode_rewards) > 1:
            total_num_steps = (i + 1) * config.num_processes * config.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(i, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))





            






            




    



    


