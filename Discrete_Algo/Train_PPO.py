import os
import glob
import time
import torch
from tqdm import tqdm
from collections import deque

from utils.PGhyperparameters import Config
from envs.ikostrov_envs import make_vec_envs, VecNormalize
from agents.PPOAgent import PPOAgent



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


#############
# Make Sure All the Data Type is cuda.Tensor
#############

if __name__ == '__main__':
    # config
    config = Config()

    # log_dir
    train_log_dir = './results/'
    exp_name = 'PPO/'
    log_dir = train_log_dir + exp_name
    makedirCleanUp(log_dir)

    # seed rng
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # Specific Configuration for PPO
    torch.set_num_threads(1)
    config.device = torch.device("cuda:0")

    config.num_steps = 128
    config.num_processes = 8
    total_epochs = int(config.num_envs_steps // config.num_steps // config.num_processes)


    # create envs which is specificed by num_processes
    env_id = 'PongNoFrameskip-v4'
    envs = make_vec_envs(env_id, config.seed, config.num_processes, log_dir, config.device, False) # allow_early_resets = False
    

    ppoAgent = PPOAgent(config, envs) # agent contains buffer like rolloutstorage

    # Predefined Variable For Training Loop
    obs = envs.reset() # The envs have been wrapped by VecPytorch
    ppoAgent.rollouts.obs[0].copy_(obs)
    ep_rews = deque(maxlen = 10)

    start = time.time()

    for epoch in tqdm(range(total_epochs)):
        # prepare_exploration_strategy_for_epochs()    (ignored)
        for step in range(config.num_steps):
            # get_action with exploration_strategy_for_steps()  (implemented in get_action)
            values, actions, action_log_probs = ppoAgent.get_action(ppoAgent.rollouts.obs[step]) 

            # excute action in envs
            obs, rewards, dones, infos = envs.step(actions)  # (1, num_processes, *shape)

            # log ep_rews info
            for info in infos:
                if 'episode' in info.keys():
                    ep_rews.append(info['episode']['r'])

            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in dones])  # One Done For Observations
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])

            ppoAgent.rollouts.store(obs, acts, rewards, action_log_probs, value_preds, masks, bad_masks)
        
        # update : num_mini_batch * mini_batch_size = config.num_processes * config.num_steps (aka config.rollouts_size) 
        value_loss, policy_loss, dist_entropy = ppoAgent.update()
        

        # reset buffer which is written into ppoAgent.update()
    
        # log the info you want for every epoch


            




