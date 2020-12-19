# nbit-6, with truncation, one figure, all T.
# with smooth

import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

smooth = True
smooth_index = 150 # [0, max)
threshold_len = 0 # == 10000 ?????????? if (threshold_len>10000): threshold <- 10000
linewidth = 1.0

def plot_func_2(dirs, color, label):
    if (smooth):
        
        threshold_list = list()
        for dir in dirs:
            event = EventAccumulator(dir)
            event.Reload()
            threshold = event.scalars.Items('threshold')
            threshold_len = len(threshold)
            print("current_dir: ", dir)
            print("threshold_len: ", threshold_len)
            tmp_threshold = np.zeros(threshold_len)
            for i in range(len(threshold)): # average with the behind
                # threshold.value = threshold[i:i+smooth_index].value
                # print("(threshold[i].step, threshold[i].value): ", (threshold[i].step, threshold[i].value))
                tmp_threshold[i] = (np.array([j.value for j in threshold[i:i+smooth_index]])).mean()
            # !!! threshold and tmp_threshold
            # thresholds = np.stack((thresholds, tmp_threshold), axis = 0)
            threshold_list.append(tmp_threshold)

            print("len(tmp_threshold): ", len(tmp_threshold))
            print("threshold[-1]: ", threshold[-1])

        # minimum threshold length of all thresholds from one T and different seeds: 
        min_threshold_len = min([len(threshold_list[i]) for i in range(len(threshold_list))])
        print("min_threshold_len: ", min_threshold_len)

        thresholds = threshold_list[0][0:min_threshold_len]
        for i in range(1, len(threshold_list)):
            print("len(thresholds): ", len(thresholds))
            print("len(threshold_list[i][0:min_threshold_len]): ", len(threshold_list[i][0:min_threshold_len]))
            thresholds = np.stack((thresholds, threshold_list[i][0:min_threshold_len]), axis = 0)
        

        print("len(thresholds): ", len(thresholds))
        # print("len(thresholds[0]): ", len(thresholds[0]))
        # plt.plot([i.step for i in threshold], tmp_threshold, color=color, label=label, linewidth=linewidth) 
    else:
        event = EventAccumulator(dir)
        event.Reload()
        threshold = event.scalars.Items('threshold')
        plt.plot([i.step for i in threshold], [i.value for i in threshold], color=color, label=label, linewidth=linewidth)
        return

def plot_func(dir, color, label):
    if (smooth):
        event = EventAccumulator(dir)
        event.Reload()
        threshold = event.scalars.Items('threshold')
        threshold_len = len(threshold)
        print("current_dir: ", dir)
        print("threshold_len: ", threshold_len)
        tmp_threshold = np.zeros(threshold_len)
        for i in range(len(threshold)): # average with the behind
            # threshold.value = threshold[i:i+smooth_index].value
            # print("(threshold[i].step, threshold[i].value): ", (threshold[i].step, threshold[i].value))
            tmp_threshold[i] = (np.array([j.value for j in threshold[i:i+smooth_index]])).mean()
        # !!! threshold and tmp_threshold
        print("tmp_threshold: ", tmp_threshold)
        plt.plot([i.step for i in threshold], tmp_threshold, color=color, label=label, linewidth=linewidth) 
    else:
        event = EventAccumulator(dir)
        event.Reload()
        threshold = event.scalars.Items('threshold')
        plt.plot([i.step for i in threshold], [i.value for i in threshold], color=color, label=label, linewidth=linewidth)
        return

#-------------------------------------------------------------------------------------
# dir func:
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

plot_func_2(final_path[0], 'r', "testtest")

#-------------------------------------------------------------------------------------

dir = "/home/pami/ZeroRL/nqubit/results/sac_energy_new/nbit-5/T-1.450seed-5/events.out.tfevents.1608019729.pami12"
event = EventAccumulator(dir)
event.Reload()
print("\n\n")
print("event: ", event)
print("event.scalars.Keys(): ", event.scalars.Keys())

# fig = plt.figure(figsize=(6,4))
# ax1 = fig.add_subplot(211)
# ax1.plot([i.step for i in threshold], [i.value for i in threshold], color='r', label='T=3.0')
# ax1.set_xlim(0)
# # acc=ea.scalars.Items('acc')
# # ax1.plot([i.step for i in acc],[i.value for i in acc],label='acc')
# ax1.set_xlabel("step")
# ax1.set_ylabel("threshold")

# dir2 = "/home/pami/ZeroRL/nqubit/results/sac_energy_new/nbit-6/T-3.500/events.out.tfevents.1607686863.pami12"
# event2 = EventAccumulator(dir2)
# event2.Reload()
# threshold2 = event2.scalars.Items('threshold')
# ax2 = fig.add_subplot(212)
# ax2.plot([i.step for i in threshold], [i.value for i in threshold], color='b', label='T=3.5')

plot_func(dir=dir, color='maroon', label='T=3.0')

# acc=ea.scalars.Items('acc')
# ax1.plot([i.step for i in acc],[i.value for i in acc],label='acc')


# dir2 = "/home/pami/ZeroRL/nqubit/results/sac_energy_new/nbit-5/T-1.450seed-6/events.out.tfevents.1608019741.pami12"
# plot_func(dir=dir2, color='steelblue', label='T=3.5')

# dir3 = "/home/pami/ZeroRL/nqubit/results/sac_energy_new/nbit-6/T-4.000/events.out.tfevents.1607686837.pami12"
# plot_func(dir=dir3, color='g', label='T=4.0')

# dir4 = "/home/pami/ZeroRL/nqubit/results/sac_energy_new/nbit-6/T-4.500/events.out.tfevents.1607686906.pami12"
# plot_func(dir=dir4, color='gold', label='T=4.5')

# dir5 = "/home/pami/ZeroRL/nqubit/results/sac_energy_new/nbit-6/T-5.000/events.out.tfevents.1607676571.pami12"
# plot_func(dir=dir5, color='grey', label='T=5.0')

# dir6 = "/home/pami/ZeroRL/nqubit/results/sac_energy_new/nbit-6/T-7.000/events.out.tfevents.1607676668.pami12"
# plot_func(dir=dir6, color='sandybrown', label='T=7.0')

# dir7 = "/home/pami/ZeroRL/nqubit/results/sac_energy_new/nbit-6/T-9.000/events.out.tfevents.1607676748.pami12"
# plot_func(dir=dir7, color='pink', label='T=9.0')

plt.xlim(0)
plt.xlabel("step")
plt.ylabel("threshold")

plt.legend(loc='lower right')
plt.savefig('original.jpg')

plt.xlim(0, 15000)
plt.savefig('truncation_x.jpg')

plt.xlim(0, 10000)
plt.savefig('truncation_8000_.jpg')

plt.xlim(0, 8000)
plt.savefig('truncation_8000_.jpg')

plt.xlim(0, 3000)
plt.savefig('truncation_3000_.jpg')

plt.xlim(0, 1000)
plt.savefig('truncation_1000_.jpg')





