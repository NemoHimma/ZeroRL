import argparse

def get_args():
    
    parser = argparse.ArgumentParser(description = 'SAC parameters for nqubit')

    # Env Specification
    parser.add_argument('--nbit', type=int, default=5)
    parser.add_argument('--T', type=float, default=1.63)
    parser.add_argument('--GPU', type=int, default=0)

    #parser.add_argument('--env_id', type=str, default='OneHotActionEnv')
    #parser.add_argument('--env_id', type=str, default='NoOneHotEnv')
    #parser.add_argument('--env_id', type=str, default='DoubleOneHotEnv')
    parser.add_argument('--env_id', type=str, default='OneHotEnv')

    # Name
    parser.add_argument('--name', type=str)
    
    # seed
    parser.add_argument('--seed', type = int, default = 1)

    # Control Variable
    parser.add_argument('--num_episodes', type = int, default = int(1e4))        # int(1e4)
    parser.add_argument('--episode_length', type = int, default = 30)            # 30 for one-hot
    parser.add_argument('--random_steps', type = int, default = 2000)            # 2000
    parser.add_argument('--learn_start_steps', type = int, default = 2000)       # 2000
    parser.add_argument('--update_freq_steps', type=int, default= 1)             # 1
    parser.add_argument('--target_update_freq', type=int, default = 2)           # 2
    parser.add_argument('--measure_every_n_steps',type=int, default = 5)         # 1

    # update related 
    parser.add_argument('--batch_size', type = int, default = 128)                # 128
    parser.add_argument('--policy_lr', type = float, default = 3e-4)
    parser.add_argument('--value_lr', type = float, default = 3e-4)
    parser.add_argument('--alpha_lr', type=float, default = 5e-5)
    parser.add_argument('--gamma', type = float, default = 0.99)
    parser.add_argument('--alpha', type = float, default = 0.02)                  # 0.02 or autotune 
    parser.add_argument('--polyak', type = float, default = 0.995)
    parser.add_argument('--update_freq_per_step', type = int, default = 2)
    parser.add_argument('--policy_decay', type = int, default = 2)

    # buffer_size
    parser.add_argument('--buffer_size', type = int, default = int(1e6)) 

    # Network Related
    parser.add_argument('--actor_hidden_size', type = int, default = 256)  
    parser.add_argument('--critic_hidden_size', type = int, default = 256)
    parser.add_argument('--actor_log_std_min', type = int, default = -10)
    parser.add_argument('--actor_log_std_max', type = int, default = 1)
    

    # Log
    parser.add_argument('--log_state_action_steps', type = int, default = 100)

    # Test
    parser.add_argument('--n_initial_points', type=int, default=3)

    # Tricks
    parser.add_argument('--auto_tune_alpha', type = bool , default = False)
    parser.add_argument('--reward_scale', type = float, default= 5.0)


    args = parser.parse_args()

    return args
