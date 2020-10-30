import argparse

def get_args():

    parser = argparse.ArgumentParser(description='DDPG, TD3, SAC parameters info')
    # device 
    parser.add_argument('--cuda', type = bool, default='True', help = 'train on CUDA device')
    # env
    parser.add_argument('--env_id', default = 'Ant-v2')
    # algo
    parser.add_argument('--algo', default = 'SAC', help='only DDPG, TD3, SAC Implemented')
    # train or eval
    parser.add_argument('--eval', type = bool, default = False)
    # seed
    parser.add_argument('--seed', type = int, default = 123456)

    # Train Procedure Variable
    parser.add_argument('--num_episodes', type = int, default = 2000)  
    parser.add_argument('--max_episodes_length', type = int, default = 2000)  # This is suitable for algo like PPO

    parser.add_argument('--total_env_steps', type = int, default = int(1e6), help = 'Training Loop index from 0 ~ total_env_steps-1') # This is suitable for SAC


    parser.add_argument('--learn_start_steps', type = int, default = 5000, help = 'model updates after learn_start_steps')
    parser.add_argument('--random_steps', type = int, default = 10000, help = 'randomly explore the space before exploiting the model')

    parser.add_argument('--update_every_steps', type = int, default =20, help = 'how many steps to update')
    parser.add_argument('--updates_per_step', type = int, default = 10, help = 'model updates per env_step')
    parser.add_argument('--policy_delay', type = int, default =2, help = 'Decay the policy update compared to value fucntion update')
    parser.add_argument('--target_update_freq', type = int, default = 1, help = 'target_model (hard or soft) update frequencies')
    

    # buffer
    parser.add_argument('--buffer_size', type = int, default = int(1e6))

    # model parameters
    parser.add_argument('--hidden_size', type = int, default = 256)
    parser.add_argument('--policy_structure', type = str, default = 'Gaussian', help = 'Gaussian or Deterministic')

    # optimization parameters
    parser.add_argument('--value_lr', type = float, default = 3e-4, help = 'critic_learning_rate')
    parser.add_argument('--policy_lr', type = float, default = 3e-4, help = 'actor_learning_rate')
    parser.add_argument('--polyak', type = float, default = 0.995, help = 'tau = 1 - polyak is used for soft update or hard update(polyak = 1)')
    parser.add_argument('--batch_size', type = int, default = 128, help = 'update_parameters')
    
    # loss function parameters
    parser.add_argument('--gamma', type = float, default = 0.99, help = 'discount factor which is needed to calculate the loss fucntion')
    parser.add_argument('--alpha', type = float, default = 0.2, help = 'coefficient of the entropy term in loss function')

    args = parser.parse_args()

    assert args.algo in ['DDPG', 'TD3', 'SAC']

    return args

# Test
if __name__ == '__main__':
    args = get_args()
    args.env_id = 'None'
    print(args.env_id)








    
