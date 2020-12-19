import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

c = np.zeros(0)
a = np.array([1, 2, 3])
b = np.array([5, 6, 7])

print(np.concatenate((a, b), axis = 0))
print(np.stack((c, b), axis = 0))
print(np.mean((a, b), axis = 0))
print(np.std((a, b), axis = 0))

thresholds = np.zeros(0)
print(thresholds)