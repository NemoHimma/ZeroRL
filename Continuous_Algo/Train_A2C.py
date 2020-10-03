import torch
import os, glob
from collections import deque
import time

from utils.hyperparameters import Config
from envs.PrepareMujoco import PrepareParallelEnv
from agents.A2CAgent import A2CAgent
from buffer.rollout import RolloutStorage

def makedirCleanUp(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


if __name__ == '__main__':
    # parameters config
    config = Config()

    # seed rng in CPU & GPU
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    # appoint log_dir & create
    train_log_dir = './A2C/train/'
    eval_log_dir = './A2C/eval/'
    makedirCleanUp(train_log_dir)
    makedirCleanUp(eval_log_dir)

    # appoint num_thread & device
    torch.set_num_threads(1)
    config.device = torch.device("cuda:0")

    # create envs
    env_id = "Ant-v2"
    envs = PrepareParallelEnv(env_id, config.seed, config.num_processes, config.gamma, train_log_dir, config.device, False)

    # Agent
    a2cAgent = A2CAgent(config, envs)

    # rollots buffer
    rollouts = RolloutStorage(config.num_steps, config.num_processes, envs.observation_space.shape, envs.action_space)


    # Initialization
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)  # has stored initial obs
    rollouts.to(config.device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    
    for i in range(config.num_updates):
        # Decay the Learning Rate If Necessary
        if config.USE_DECAY_LR:
            config.LinearDecayLR(a2cAgent.optimizer, i , config.num_updates, config.lr)
        
        for step in range(config.num_steps):
            # sample action
            with torch.no_grad():
                values, action, action_log_probs = a2cAgent.get_action(obs[step])

            # excute action in envs
            obs, rewards, done, infos = envs.step(action)
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
            
            # process masks & bad_masks
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            
            # store(obs, actions, action_log_probs, values, rewards, masks, bad_masks)
            rollouts.insert(obs, action, action_log_probs, values, rewards, masks, bad_masks)

        # deals with truncated epsiode
        with torch.no_grad():
            next_value = a2cAgent.get_critic_value(rollouts.obs[-1]).detach()

        # conpute return
        rollouts.compute_returns(next_value, config.USE_GAE, config.gamma, config.gae_lambda, config.USE_PROPER_TIME_LIMITS)

        # update
        value_loss, action_loss, dist_entropy = a2cAgent.update(rollouts)

        # reset rollouts
        rollouts.after_update()


        ################ Save & Log #########################





            




    



    


