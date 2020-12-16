import argparse

def get_args():
    
    parser = argparse.ArgumentParser(description = 'SAC parameters for nqubit')

    # Env Specification
    parser.add_argument('--nbit', type=int, default=5)
    parser.add_argument('--T', type=float, default=1.63)
    parser.add_argument('--GPU', type=int, default=0)
    
    # seed
    parser.add_argument('--seed', type = int, default = 1)

    # Control Variable
    parser.add_argument('--num_episodes', type = int, default = int(1e6))
    parser.add_argument('--max_episode_steps', type = int, default = 3) # 3
    parser.add_argument('--start_to_exploit_steps', type = int, default = 1200) # 900
    parser.add_argument('--learn_start_steps', type = int, default = 1200) # 900

    # update related 
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--policy_lr', type = float, default = 3e-4)
    parser.add_argument('--value_lr', type = float, default = 3e-4)
    parser.add_argument('--gamma', type = float, default = 0.99)
    parser.add_argument('--alpha', type = float, default = 0.02)
    parser.add_argument('--polyak', type = float, default = 0.995)
    parser.add_argument('--update_freq_per_step', type = int, default = 10)
    parser.add_argument('--policy_decay', type = int, default = 5)

    # buffer_size
    parser.add_argument('--buffer_size', type = int, default = int(1e6))

    # Network Related
    parser.add_argument('--actor_hidden_size', type = int, default = 64)  # 64 
    parser.add_argument('--critic_hidden_size', type = int, default = 256)
    parser.add_argument('--actor_log_std_min', type = int, default = -20)
    parser.add_argument('--actor_log_std_max', type = int, default = -4)

    # Log
    parser.add_argument('--log_state_action_steps', type = int, default = 100)


    args = parser.parse_args()

    return args
    

def get_dqn_args():
    parser = argparse.ArgumentParser(description = 'SAC parameters for nqubit')


if __name__ == "__main__":
    args = get_args()
    args_dict = vars(args)
    print(type(args_dict))

