import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


appointed_folder = "nbit-5"

original_path = "../results/sac_energy_new/" + appointed_folder + "/*"
print(original_path)
paths = glob.glob(original_path) # type:string
print("paths: ", paths)

prog = re.compile('T-[^s]*')
print("prog.search(paths[0]): ", prog.search(paths[0]))
T_value_set = set()

for i in range(len(paths)):
    T_value_set.add(prog.search(paths[i]).group())
print("T_value_set: ", T_value_set)
final_path = list()
T_value_list = list(T_value_set)

for i in range(len(T_value_list)):
    final_path.append(list())
    prog = re.compile(T_value_list[i])
    for j in range(len(paths)):
        if (prog.search(paths[j]) != None):
            final_path[i].append(paths[j])

print("final_path: ", final_path)

event = EventAccumulator(final_path[0][0])
event.Reload()
threshold = event.scalars.Items('threshold')
