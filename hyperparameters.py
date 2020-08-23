import torch 
import math

class Config(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tricks
        self.USE_NOISY_NETS = False
        self.USE_PRIORITY_REPLAY = False
        self.N_STEPS = 1

        # Noisy_Net
        self.SIGMA_INIT = 0.5

        # Priority Replay Parameters
        self.PRIORITY_ALPHA = 0.6
        self.PRIORITY_BETA_START = 0.4
        self.PRIORITY_BETA_FRAMES = 100000

        # N_STEPS Discount Factor
        self.GAMMA = 0.99

        # Categorical Parameters
        self.ATOMS = 51
        self.V_MAX = 10
        self.V_MIN = -10

        # QR_DQN Parameters
        self.QUANTILES = 51

        # DR_DQN Parameters
        self.SEQUENCY_LENGTH = 8

        # Epsilon_greedy parameters

        '''
        When random number is greater than epilon, it exploits the model. Otherwise, it explores.
        '''

        self.epison_start = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 30000

        self.epsilon_by_frame = lambda frame_idx:self.epsilon_final + (self.epison_start - self.epsilon_final) * math.exp(-1. * frame_idx / self.epsilon_decay)

        # Learning Rate & Batch Size
        self.LR = 1e-4
        self.BATCH_SIZE = 32

        # Buffer Size
        self.Replay_Buffer_Size = 100000

        # Learning Procedure Parameters
        self.TARGET_NET_UPDATE_FREQUENCY = 1000
        self.CURRENT_NET_UPDATE_FREQUENCY = 1
        self.LEARN_START = 10000   # learn_start
        self.MAX_FRAMES = 100000   # learn_end

        # data logging parameters
        self.ACTION_SELECTION_COUNT_FREQUENCY = 1000