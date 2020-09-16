import torch
import torch.optim as optim
import csv

from buffer.ReplayBuffer import ExperienceReplayBuffer
from networks.networks import DQN
from networks.bodies import AtariBody

class DQNAgent(object):
    def __init__(self, config=None, env=None, log_dir=None, static_policy=False):
        # Train or Test
        self.static_policy = static_policy

        # Tricks Flags
        self.noisy = config.USE_NOISY_NETS
        self.priority_replay = config.USE_PRIORITY_REPLAY
        self.nsteps = config.N_STEPS

        # Tricks Parameters
        self.sigma_init = config.SIGMA_INIT
        self.alpha = config.PRIORITY_ALPHA
        self.priority_beta_start = config.PRIORITY_BETA_START
        self.priority_beta_frames = config.PRIORITY_BETA_FRAMES
        self.nstep_buffer = []

        # Categorical-DQN
        self.ATOMS = config.ATOMS
        self.V_MAX = config.V_MAX
        self.V_MIN = config.V_MIN

        # QR-DQN
        self.quantiles =config.QUANTILES

        # Device
        self.device = config.device
        
        # Memory
        self.replay_buffer_size = config.REPLAY_BUFFER_SIZE

        # LR & BATCH_SIZE & Discount
        self.lr = config.LR
        self.batch_size = config.BATCH_SIZE
        self.gamma = config.GAMMA

        # Learn Procedure
        self.max_frames = config.MAX_FRAMES
        self.learn_start = config.LEARN_START
        self.update_freq = config.UPDATE_FREQ
        self.target_update_freq = config.TARGET_UPDATE_FREQ
        self.update_count = 0

        # Log Info
        self.action_log_frequency = config.ACTION_SELECTION_COUNT_FREQUENCY
        self.log_dir = log_dir
        self.rewards = []
        self.action_selections = [0 for _ in range(env.action_space.n)]

        # Exploration_policy
        self.epsilon_start = config.EPSILON_START
        self.epsilon_final = config.EPSILON_FINAL
        self.epsilon_decay = config.EPSILON_DECAY
        self.epsilon_by_frame = config.EPSILON_BY_FRAME

        # Env 
        self.env = env
        self.input_shape = env.observation_space.shape
        self.num_actions = env.action_space.n

        # Construct Entities
        self.declare_memory()
        self.declare_networks()

        # Network Initialization & Optimizer & Movement
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)


        # Train or Test Function
        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()

    def declare_memory(self):
        if not self.priority_replay:
            self.memory = ExperienceReplayBuffer(self.replay_buffer_size)
        else:
            pass
    
    def declare_networks(self):
        self.model = DQN(self.input_shape, self.num_actions, self.noisy, self.sigma_init, body = AtariBody)
        self.target_model = DQN(self.input_shape, self.num_actions, self.noisy, self.sigma_init, body = AtariBody)

    def loss_functon(self, loss_name, x):

        if loss_name == 'huber':
            cond = (x.abs()< 1.0).float().detach()
            return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)

        if loss_name == 'MSE':
            return 0.5 * x.pow(2)

###################### Main Update Step #######################
    def update(self, s, a, r, s_, frame_idx):
        # train or test
        if self.static_policy:
            return None

        # store in memroy
        self.append_to_replay(s,a,r,s_)

        # when to learn & how often we learn
        if frame_idx < self.learn_start or frame_idx % self.update_freq !=0:
            return None

        # prepare_data from memory
        batch_vars = self.prep_minibatch()

        # compute the task-specific loss
        loss = self.compute_loss(batch_vars)

        # gradient update
        self.optimizer.zero_grad() 
        loss.backward()
        for parameter in self.model.parameters():
            parameter.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # when to update the target
        self.update_target_model()

        # save some info like the TD-error
        self.save_td(loss.item(), frame_idx)
        self.save_sigma_param_magnitudes(frame_idx)
        
############################### Utility Function ################
    def append_to_replay(self,s ,a, r, s_):
        self.nstep_buffer.append((s,a, r, s_))
        if (len(self.nstep_buffer)< self.nsteps):
            return
        
        R = sum([self.nstep_buffer[i][2] * (self.gamma ** i) for i in range(self.nsteps)])
        state, action, _, _ = self.nstep_buffer.pop(0)
        self.memory.push((state, action, R, s_))        

    def prep_minibatch(self):
        transitions, indices, weights = self.memory.sample(self.batch_size)
        batch_states, batch_actions, batch_rewards, batch_next_states = zip(*transitions)
        
        batch_states = torch.tensor(batch_states, device = self.device, dtype = torch.float).view(batch_states_shape)
        batch_actions = torch.tensor(batch_actions, device = self.device, dtype = torch.long).view(-1, 1)
        batch_rewards = torch.tensor(batch_rewards, device = self.device, dtype = torch.float).view(-1, 1)

        non_final_mask = torch.tensor(tuple(map(lambda s : s is not None, batch_next_states)), device = self.device, dtype = torch.uint8)

        try: #sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float).view(batch_states_shape)
            empty_next_state_values = False

        except:
            non_final_next_states = None
            empty_next_state_values = True

        return batch_states, batch_actions, batch_rewards, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights

    def compute_loss(self):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars
        
        # current-q-values
        self.model.sample_noise()
        current_q_values = self.model(batch_state).gather(1, batch_action)

        # target-q-values
        with torch.no_grad():
            max_next_q_values = torch.zeros(self.batch_size, device = self.device, dtype = torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                # get_max_next_state_action
                max_next_action = self.get_max_next_state_action(non_final_next_states)  # action selection comes from target model (not double)
                self.target_model.sample_noise()
                max_next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_state_action)

            target_q_values = batch_reward + (self.gamma ** self.nsteps) * max_next_q_values
        
        diff = target_q_values - current_q_values
        loss = self.loss_functon('MSE', diff)
        loss = loss.mean()

        return loss

    def get_max_next_state_action(self, next_states):
        max_next_action = self.target_model(non_final_next_states).max(dim=1)[1].view(-1, 1)
        return max_next_action

    def update_target_model(self):
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            self.update_count = 0

    def get_action(self, s, eps):
        with torch.no_grad():
            if np.random.random() >= eps or self.static_policy or self.noisy:
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                self.model.sample_noise()
                a = self.model(X).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    def finish_nstep(self):
        while len(self.nstep_buffer) > 0:
            R = sum([self.nstep_buffer[i][2]*(self.gamma**i) for i in range(len(self.nstep_buffer))])
            state, action, _, _ = self.nstep_buffer.pop(0)

            self.memory.push((state, action, R, None))               

################################ save & load & log ############################

    def save_weight(self):
        torch.save(self.model.state_dict(), os.path.join(self.log_dir,'model.dump'))
        torch.save(self.optimizer.state_dict(), os.path.join(self.log_dir, 'optim.dump'))

    def load_weight(self):
        fname_model = os.path.join(self.log_dir, 'model.dump')
        fname_optim = os.path.join(self.log_dir,'optim.dump')

        if os.path.isfile(fname_model):
            self.model.load_state_dict(torch.load(fname_model))
            self.target_model.load_state_dict(self.model.state_dict())

        if os.path.isfile(fname_optim):
            self.optimizer.load_state_dict(torch.load(fname_optim))

    def save_replay(self):
        pickle.dump(self.memory, open(os.path.join(self.log_dir, 'exp_replay_agent.dump'), 'wb'))

    def load_replay(self):
        fname = os.path.join(self.log_dir, 'exp_replay_agent.dump')
        if os.path.isfile(fname):
            self.memory = pickle.load(open(fname, 'rb'))

    def save_td(self, loss, timestep):
        with open(os.path.join(self.log_dir, 'td.csv'),'a') as f:
            writer = csv.writer(f)
            writer.writerow((timestep, loss))

    def save_reward(self, episode_reward):
        self.rewards.append(episode_reward)

    def save_action(self, action, tstep):
        self.action_selections[int(action)] += 1.0/self.action_log_frequency
        if (tstep+1) % self.action_log_frequency == 0:
            with open(os.path.join(self.log_dir, 'action_log.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(list([tstep]+self.action_selections))
            self.action_selections = [0 for _ in range(len(self.action_selections))]   

    def save_sigma_param_magnitudes(self, tstep):
        with torch.no_grad():
            sum_, count = 0.0, 0.0
            for name, param in self.model.named_parameters():
                if param.requires_grad and 'sigma' in name:
                    sum_+= torch.sum(param.abs()).item()
                    count += np.prod(param.shape)
            
            if count > 0:
                with open(os.path.join(self.log_dir, 'sig_param_mag.csv'), 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow((tstep, sum_/count)) 



    

            


        
