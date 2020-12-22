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
stack_a_b = np.stack((a, b), axis = 0)
print("stack_a_b:\n ", stack_a_b)
print("stack_a_b.mean():\n ", stack_a_b.mean(axis = 0))
print(np.mean((a, b), axis = 0))
print(np.std((a, b), axis = 0))

print("b[3:4]: ", b[1:1])

thresholds = np.zeros(0)
print(thresholds)