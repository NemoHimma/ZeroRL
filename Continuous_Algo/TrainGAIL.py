import os
import glob
import time
import numpy as np
from collections import deque

import torch
import argparse

# envs
from envs.ikostrov_envs import make_vec_envs
# buffer
from buffer.storage import RolloutStorage
########## args ###########
def get_args():
    parser = argparse.ArgumentParser(description='PPO Policy Generator & Binary Discriminator')

    # device 
    parser.add_argument('--cuda', type = bool, default='True', help = 'train on CUDA device')
    # env
    parser.add_argument('--env_id', default = 'HalfCheetah-v2')  
    # seed
    parser.add_argument('--seed', type = int, default = 123456)
    # num_processes
    parser.add_argument('--num_processes', type = int, default = 8)
    # num_steps
    parser.add_argument('--num_steps', type = int, default = 64)
    # total_env_steps
    parser.add_argument('--total_env_steps', type = int, default = int(1e7))
    # algo_name
    parser.add_argument('--algo', type = str, default = 'gail')

    # gamma
    parser.add_argument('--gamma', type = float, default = 0.99)
    # gail_epoch
    parser.add_argument('--gail_epoch', type = int, default = 5)
    # use_gae
    parser.add_argument('--use_gae', type = bool, default = True)
    # proper_time_limits
    parser.add_argument('--proper_time_limits', type = bool, default = True)
    # gae_lambda
    parser.add_argument('--gae_lambda', type = float, default = 0.95)




    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    # specific configuration
    args.seed = 123456
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # seed 
    torch.manual_seed(args.seed)
    torch.manual_seed_all(args.seed)

    #log_dir
    prefix = './results/'
    log_dir = os.path.join(prefix , args.env_id + '/' + args.algo)

    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.pami12')) + glob.glob(os.path.join(log_dir, '*.dump'))
        for f in files:
            os.remove(f)

    # tensorboardX
    writer = SummaryWriter(log_dir)

    # Parallel Envs (VectorPytorch)
    envs = make_vec_envs(args.env_id, args.seed, args.num_processes, args.gamma, log_dir, device, False)
    
    # RolloutBuffer
    rollouts = RolloutStorage(args.num_steps, args.num_processes, envs.observation_space.shape, envs.action_space, 1)

    # Agent (G:policy, D:Discriminator)
    agent = GailAgent()

    #  Training Loop
    obs = envs.reset() # (1, num_processes, obs_shape)
    rollots.obs[0].copy_(obs)
    rollots.to(device)
    episode_rewards = deque(maxlen=10)
    start = time.time()

    num_updates = int(args.total_env_steps // args.num_steps // args.num_processes)

    for update in tqdm(range(num_updates)):
        # Linear Decay LR If Necessary

        # policy num_steps interaction
        for step in range(args.num_steps):
            action , action_log_probs, value  = agent.get_action(rollouts.obs[step], rollouts.masks[step])

            obs, reward, done, infos = envs.step(action)

            # real done mask
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            # time_limit_end mask
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

            rollouts.insert(obs, obs, action, action_log_probs, value, reward, masks, bad_masks)
        
        # compute next_value for the rollouts designed buffer
        with torch.no_grad():
            next_value = agent.get_value(rollouts.obs[-1], rollouts.masks[-1]).detach()

        # Gail Update Part
        if update < 10:
            gail_epoch = 100
        else:
            envs.venv.eval()
            gail_epoch = args.gail_epoch

        for _ in range(gail_epoch):
            # require expert data, policy Interaction data in rollouts, running mean std object
            agent.Discriminator.update(expert_dataset, rollouts, rms)

        # Change rollouts reward to Discriminator Output signal

        for step in range(args.num_steps):
            rollouts.rewards[step] = agent.Discriminator.predict(rollouts.obs[step], rollouts.actions[step], rollouts.masks[step], args.gamma)

        # returns for policy update
        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        # policy update
        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        # reset buffer
        rollouts.after_update()

        



    
