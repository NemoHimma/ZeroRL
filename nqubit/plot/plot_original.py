# nbit-5, with truncation, one figure, all T.
# with smooth

import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

appointed_folder = "nbit-5"
smooth = True
smooth_index = 150 # [1, +max), could be changed into 1
threshold_len = 0 # == 10000? if (threshold_len>10000): threshold <- 10000
linewidth = 1.0 # linewidth of the figure line
color_dictionary = {
    0: "#0000FF", 1: "#1F618D", 2: "#2980B9", 3: "#7FB3D5", 4: "#22DAF3",
    5: "#5B2C6F", 6: "#800000", 7: "#008000", 8: "#008000", 9: "#E74C3C",
    10: "#D35400", 11: "#800000", 12: "#E74C3C", 13: "#F1948A", 14: "#1C2833",
    15: "#F322CD", 16: "#0E0F0F"
}

def mean_std_fillcolor_plot(thresholds, color, label):
    thresholds_mean = thresholds.mean(axis = 0)
    x = [i for i in range(len(thresholds_mean))]
    thresholds_std = thresholds.std(axis = 0)
    superbound = thresholds_mean + thresholds_std
    lowerbound = thresholds_mean - thresholds_std

    plt.plot(x, thresholds_mean, color=color, label=label, linewidth=linewidth) 
    plt.fill_between(x, superbound, lowerbound, where=superbound>=lowerbound, facecolor=color, interpolate=True, alpha=0.1)
    return

def smooth_func(threshold, tmp_threshold):
    for i in range(len(threshold)): # average with the behind
        # threshold.value = threshold[i:i+smooth_index].value
        # print("(threshold[i].step, threshold[i].value): ", (threshold[i].step, threshold[i].value))
        tmp_threshold[i] = (np.array([j.value for j in threshold[i:i+smooth_index]])).mean()
    return tmp_threshold

def plot_func(dirs, color, label):
    threshold_list = list()
    for dir in dirs:
        event = EventAccumulator(dir)
        event.Reload()
        threshold = event.scalars.Items('threshold')
        threshold_len = len(threshold)
        tmp_threshold = np.zeros(threshold_len)
        if (not smooth):
            smooth_index = 1
        tmp_threshold = smooth_func(threshold, tmp_threshold)
        threshold_list.append(tmp_threshold)

    # minimum threshold length of all thresholds from one T and different seeds: 
    min_threshold_len = min([len(threshold_list[i]) for i in range(len(threshold_list))])
    thresholds = np.zeros(0)
    for i in range(0, len(threshold_list)):
        thresholds = np.concatenate((thresholds, threshold_list[i][0:min_threshold_len]), axis = 0)
    thresholds = thresholds.reshape((len(threshold_list), min_threshold_len))
    mean_std_fillcolor_plot(thresholds, color, label)
    # plt.plot([i.step for i in threshold], tmp_threshold, color=color, label=label, linewidth=linewidth) 

#-------------------------------------------------------------------------------------
# dir func:
original_path = "../results/sac_energy_new/" + appointed_folder + "/*"
paths = glob.glob(original_path) # paths type:string list

tar_re = re.compile('T-[^s]*')
T_value_set = set()

for i in range(len(paths)):
    T_value_set.add(tar_re.search(paths[i]).group())
final_path = list()
T_value_list = list(T_value_set)

for i in range(len(T_value_list)):
    final_path.append(list())
    tar_re = re.compile(T_value_list[i])
    for j in range(len(paths)):
        if (tar_re.search(paths[j]) != None):
            final_path[i].append(paths[j])

for i in range(len(final_path)):
    plot_func(final_path[i], color_dictionary[i], T_value_list[i])
#-------------------------------------------------------------------------------------

plt.xlim(0)
plt.xlabel("step")
plt.ylabel("threshold")

# plt.legend(loc='lower right')
plt.legend(bbox_to_anchor=(0.5, -0.3), loc="lower center")
plt.title(appointed_folder)
plt.savefig('original.jpg', dpi=600, bbox_inches='tight')

plt.xlim(0, 15000)
plt.legend(bbox_to_anchor=(0.5, -0.3), loc="lower center")
plt.savefig('truncation_x.jpg')

plt.xlim(0, 10000)
plt.legend(bbox_to_anchor=(0.5, -0.3), loc="lower center")
plt.savefig('truncation_8000_.jpg')

plt.xlim(0, 8000)
plt.legend(bbox_to_anchor=(0.5, -0.3), loc="lower center")
plt.savefig('truncation_8000_.jpg')

plt.xlim(0, 3000)
plt.legend(bbox_to_anchor=(0.5, -0.3), loc="lower center")
plt.savefig('truncation_3000_.jpg')

plt.xlim(0, 1000)
plt.legend(bbox_to_anchor=(0.5, -0.3), loc="lower center")
plt.savefig('truncation_1000_.jpg')





