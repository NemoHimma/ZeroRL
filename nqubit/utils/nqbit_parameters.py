import argparse

def get_args():
    
    parser = argparse.ArgumentParser(description = 'SAC parameters for nqubit')

    # Env Specification
    parser.add_argument('--nbit', type=int, default=5)
    parser.add_argument('--T', type=float, default=1.63)
    parser.add_argument('--GPU', type=int, default=0)

    # Name
    parser.add_argument('--name', type=str)
    
    # seed
    parser.add_argument('--seed', type = int, default = 1)

    # Control Variable
    parser.add_argument('--num_episodes', type = int, default = 1000)
    parser.add_argument('--episode_length', type = int, default = 100) # 3
    parser.add_argument('--random_steps', type = int, default = 200) # 900
    parser.add_argument('--learn_start_steps', type = int, default = 300) # 900
    parser.add_argument('--update_freq_steps', type=int, default= 200)
    parser.add_argument('--target_update_freq', type=int, default = 1000)
    parser.add_argument('--measure_every_n_steps',type=int, default = 5)

    # update related 
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--policy_lr', type = float, default = 3e-4)
    parser.add_argument('--value_lr', type = float, default = 3e-4)
    parser.add_argument('--alpha_lr', type=float, default = 5e-5)
    parser.add_argument('--gamma', type = float, default = 0.99)
    parser.add_argument('--alpha', type = float, default = 0.2) # auto-tune 
    parser.add_argument('--polyak', type = float, default = 0.995)
    parser.add_argument('--update_freq_per_step', type = int, default = 2)
    parser.add_argument('--policy_decay', type = int, default = 2)

    # buffer_size
    parser.add_argument('--buffer_size', type = int, default = int(1e5))

    # Network Related
    parser.add_argument('--actor_hidden_size', type = int, default = 256)  # 64 
    parser.add_argument('--critic_hidden_size', type = int, default = 256)
    parser.add_argument('--actor_log_std_min', type = int, default = -9)
    parser.add_argument('--actor_log_std_max', type = int, default = 0)
    

    # Log
    parser.add_argument('--log_state_action_steps', type = int, default = 100)

    # Test
    parser.add_argument('--n_initial_points', type=int, default=3)

    # Tricks
    parser.add_argument('--auto_tune_alpha', type = bool , default = True)
    parser.add_argument('--reward_scale', type = float, default= 10.0)


    args = parser.parse_args()

    return args
    

def get_dqn_args():
    parser = argparse.ArgumentParser(description = 'DQN parameters for nqubit')
    
    # problem 
    parser.add_argument('--nbit', type=int, default=5)

    # update related
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--target_update_freq', type=int, default=50)
    parser.add_argument('--update_freq', type=int, default=2)

    # Exploration Stragedy
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_final', type=float, default=0.01)
    parser.add_argument('--epsilon_decay', type=int, default=50)
    parser.add_argument('--epsilon_rate', type=float, default=-0.1)
    parser.add_argument('--epsilon_min', type=float, default=0.05)
    
    # noisy flag
    parser.add_argument('--noisy', type=bool, default=False)
    parser.add_argument('--sigma_init', type=float, default=0.5)

    # buffer_size
    parser.add_argument('--memory_size', type=int, default=int(1e3))

    # network_hidden_size
    parser.add_argument('--hidden_size', type=int, default=20)

    # Control Variable
    parser.add_argument('--num_episodes', type = int, default=80)
    parser.add_argument('--episode_length', type=int, default=1000)
    parser.add_argument('--action_delta', type=float, default=1e-2)
    parser.add_argument('--Temp', type=float, default=10.0)
    parser.add_argument('--learn_start_steps', type=int, default=50)
    parser.add_argument('--test_freq',type=int, default=1000)
    parser.add_argument('--test_step', type=int, default=30)

    args = parser.parse_args()
    return args

def get_ddpg_args():

    parser = argparse.ArgumentParser(description = 'DDPG parameters for nqubit')

    # Env Specification
    parser.add_argument('--nbit', type=int, default=5)
    parser.add_argument('--T', type=float, default=1.63)
    parser.add_argument('--GPU', type=int, default=0)

    # Name
    parser.add_argument('--name', type=str)
    
    # seed
    parser.add_argument('--seed', type = int, default = 1)

    # Control Variable
    parser.add_argument('--num_episodes', type = int, default = 1000)
    parser.add_argument('--episode_length', type = int, default = 100) # 3
    parser.add_argument('--random_steps', type = int, default = 200) # 900
    parser.add_argument('--learn_start_steps', type = int, default = 300) # 900
    parser.add_argument('--update_freq_steps', type=int, default= 200)
    parser.add_argument('--target_update_freq', type=int, default = 1000)
    parser.add_argument('--measure_every_n_steps',type=int, default = 10)

    # update related 
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--policy_lr', type = float, default = 3e-4)
    parser.add_argument('--value_lr', type = float, default = 3e-4)
    parser.add_argument('--alpha_lr', type=float, default = 5e-5)
    parser.add_argument('--gamma', type = float, default = 0.99)
    parser.add_argument('--alpha', type = float, default = 0.2) # auto-tune 
    parser.add_argument('--polyak', type = float, default = 0.995)
    parser.add_argument('--update_freq_per_step', type = int, default = 2)
    parser.add_argument('--policy_decay', type = int, default = 2)

    # buffer_size
    parser.add_argument('--buffer_size', type = int, default = int(1e5))

    # Log
    parser.add_argument('--log_state_action_steps', type = int, default = 100)

    # Test
    parser.add_argument('--n_initial_points', type=int, default=3)

    # Tricks
    parser.add_argument('--action_noise', type = float , default = 0.01)
    parser.add_argument('--reward_scale', type = float, default= 10.0)


    args = parser.parse_args()

    return args


def get_td3_args():
    pass

    


if __name__ == "__main__":
    args = get_args()
    args_dict = vars(args)
    print(args_dict)

