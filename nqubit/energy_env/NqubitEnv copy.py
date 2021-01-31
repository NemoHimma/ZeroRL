import gym
from gym import error, spaces, utils
from gym.utils import seeding

from energy_env import HB
from energy_env import HP
from energy_env import measure

import time
import math
import random
import copy
import datetime
import numpy as np
import scipy.sparse.linalg

from sklearn.preprocessing import OneHotEncoder

# pass nbits &  Numbers to MakeMatrix
nqubits_para = {
    '5':[[49, 7, 7, 3],[57, 3, 19, 3],[69, 3, 23, 3],[87, 3, 29, 3]],
    '6':[[123, 3, 41, 3],[129, 3, 43, 3],[183, 3, 61, 3]],
    '7':[[55, 5, 11, 3],[65, 5, 13, 3],[77, 7, 11, 3],[91, 7, 13, 3]],
    '9':[[115, 5, 23, 3],[119, 7, 17, 3],[133, 7, 19, 3],[155, 5, 31, 3],[161, 7, 23, 3],[203, 7, 29, 3],[217, 7, 31, 3]],
    '10':[[121, 11, 11, 3],[169, 13, 13, 3],[291, 3, 97, 3],[339, 3, 113, 3],[381, 3, 127, 3]],
    '11':[[215, 5, 43, 3],[235, 5, 47, 3],[393, 3, 131, 3],[411, 3, 137, 3],[417, 3, 139, 3],[489, 3, 163, 3],[519, 3, 173, 3],[573, 3, 191, 3],[579, 3, 193, 3],[633, 3, 211, 3]],
    '13':[[209, 11, 19, 3],[247, 13, 19, 3],[253, 11, 23, 3]],
    '14':[[259, 7, 37, 3],[287, 7, 41, 3],[301, 7, 43, 3],[329, 7, 47, 3],[371, 7, 53, 3],[413, 7, 59, 3],[427, 7, 61, 3]]
}

Easy_evolution_time = {
    '5':1.6344, 
    '6':3.326, 
    '7':5.006, 
    '9':9.187, 
    '10':13.066,
    '11':14.384,
    '13':24.572,
    '14':28.044
}

Hard_evolution_time = {
    '5':2.5,
    '6':3.459,
    '7':6.347,
    '9':12.680,
    '10':16.257,
    '11':16.378,
    '13':32.020,
    '14':57.518
}


class NqubitEnvOneHot(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, max_episode_steps=30, nbit=5, measure_every_n_steps=1, reward_scale=5.0):
        super(NqubitEnvOneHot, self).__init__()
        

        # args
        self.reward_scale = reward_scale
        self.measure_every_n_steps = measure_every_n_steps

        # one-hot encoding
        self.enc = OneHotEncoder()
        self.evolution_step = max_episode_steps + 1
        time_label = np.reshape(np.array([i for i in range(self.evolution_step)]), (self.evolution_step, 1))
        self.enc.fit(time_label)
        
        # Env Space
        self.action_space = spaces.Box(low = -0.01, high = 0.01, shape = (6, ), dtype = np.float32)
        self.observation_space = spaces.Box(low = -1.0 , high = 1.0, shape=(self.evolution_step + 6, ), dtype = np.float32)

        # Measure
        self.nbits = nbit # n 
        self.Numbers = nqubits_para[str(self.nbits)] # Numbers
        self.g = 1e-2   # g
        self.Hb, self.Hp_array = self.MakeMatrix(self.nbits, self.Numbers) # Hb, Hp_array
        self.T = Easy_evolution_time[str(nbit)] # T 

        # Record 
        self.done = False
        self.counter = 0
        self.state = None  # s
        #self.Pi = np.pi

        #self.action_buffer = []


    def step(self, action):

        self.counter += 1

        b = self.state[self.evolution_step:] + action 
        
        time_encoding = self.enc.transform([[self.counter]]).toarray()
        self.state = np.hstack([time_encoding, np.reshape(b, (1, 6))])[0] # (1, 9) ---> (9, )
        
        if (self.counter % self.measure_every_n_steps == 0):
            measure_state = b
            reward, threshold = measure.CalcuFidelity(self.nbits, measure_state, self.Hb, self.Hp_array, self.T, self.g)

            if self.counter == (self.evolution_step - 1) :

                self.done = True

            #measure_state = np.sum(self.action_buffer, axis = 0)
            #measure_state  = b
           # reward, threshold = measure.CalcuFidelity(self.nbits, measure_state, self.Hb, self.Hp_array, self.T, self.g)

                return self.state, threshold * self.reward_scale, self.done, {'threshold':threshold, 'solution':measure_state}
        
            return self.state, threshold * self.reward_scale, self.done, {'threshold':threshold, 'solution':measure_state}
        
        return self.state, 0.0, self.done, {'measure':False}
        
    def reset(self):
        self.counter = 0
        time_encoding = self.enc.transform([[self.counter]]).toarray()  # (1,3)
        
        initial_action = np.zeros((6,) , dtype = np.float32)
        self.state = np.hstack([time_encoding, np.reshape(initial_action, (1, 6))])[0]  # (1, 9) ---> (9, )
        self.done = False
        #self.action_buffer = []
        
        return self.state

    def MakeMatrix(self, n , Numbers):
        lenthNumbers = len(Numbers)
        Hp_array  = np.zeros((lenthNumbers,2**n),dtype = float)
        Hb = HB.HB(n)

        for i in range(len(Numbers)):
            number = Numbers[i]
            fact = HP.Factorization(number[0],number[1],number[2],number[3])
            Hp_array[i][:]=fact.Hamiltonian_matrix()
        
        return Hb, Hp_array
