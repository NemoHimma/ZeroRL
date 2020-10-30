import numpy as np
import time
import HB 
import HP
import math
import scipy.sparse.linalg
import random
import copy
import datetime
import measure
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class env(object):
	def __init__(self):
		super(env, self).__init__()
		self.action_space = ['0','1+', '1-', '2+', '2-','3+','3-','4+','4-','5+','5-','6+','6-']
		self.n_actions = len(self.action_space)
		self.n_features = 6


	def step(self,n,action,s,action_delta,T,Hb,Hp_array,g):
		Pi = np.pi
		delta=action_delta
		p=copy.deepcopy(s)
		if action ==1:
			p[0] += delta
		elif action ==2:
			p[0] -= delta
		elif action ==3:
			p[1] += delta
		elif action ==4:
			p[1] -= delta
		elif action ==5:
			p[2] += delta
		elif action ==6:
			p[2] -= delta
		elif action ==7:
			p[3] += delta
		elif action ==8:
			p[3] -= delta
		elif action ==9:
			p[4] += delta
		elif action ==10:
			p[4] -= delta
		elif action ==11:
			p[5] += delta
		elif action ==12:
			p[5] -= delta

		t = np.linspace(0,T,1000)
		path = (1/T*t)+p[0]*np.sin(1*Pi*t/T)+p[1]*np.sin(2*Pi*t/T)+p[2]*np.sin(3*Pi*t/T)\
+p[3]*np.sin(4*Pi*t/T)+p[4]*np.sin(5*Pi*t/T)+p[5]*np.sin(6*Pi*t/T)
		strictly_increasing = all(x<=y for x,y in zip(path,path[1:]))
		if strictly_increasing == 0:
			done = True
			reward = measure.CalcuFidelity(n,s,Hb,Hp_array,T,g) 
		else:
			done = True
			reward = measure.CalcuFidelity(n,p,Hb,Hp_array,T,g) 
			s = np.array([p[0],p[1],p[2],p[3],p[4],p[5]])

		return s,reward,done











