import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


color_dictionary = {
    0: "#1F618D", 1: "#F322CD", 2: "#E74C3C", 3: "#000000", 4: "#22DAF3",
    5: "#5B2C6F", 6: "#800000", 7: "#008000", 8: "#008000", 9: "D35400",
    10: "#0E0F0F", 11: "#7FB3D5", 12: "#2980B9", 13: "#F1948A", 14: "#1C2833",
    15: "#E74C3C", 16: "#0000FF" 
}

marker_dictionary = {
    0:"o", 1:"^", 2:"D", 3:"x", 4:">", 5:"1", 6:"p", 7:"P", 8:"*"
}

def plot_func(ax, method_dirs, color, label, marker, smooth_index=4, alpha=0.45, linewidth=1.2, scatter_space = 400):
    '''
    input: method_dirs : ['algo1/seed1','~algo1/seed4']
    '''
    ##### extrat data from all seeds #####
    y_seeds = []
    y_length = []
    seed_count = len(method_dirs)
    
    for dir1 in method_dirs:
        
        event = EventAccumulator(dir1)
        event.Reload()
        if dir1 == '../results/latest_version9/SAC/seed1':
            y = event.scalars.Items('threshold')
        else:
            y = event.scalars.Items('episode_threshold') #  threshold_value
        y_len = len(y)                               # threshold_len
        y_length.append(y_len)
        
    ######  smoothing  #########
        smooth_array = np.zeros(y_len)
        for i in range(y_len):
            smooth_array[i] = np.array([j.value for j in y[i:i+smooth_index]]).mean()
        y_seeds.append(smooth_array)
        
    ######  reshape y_data into [num_seeds, min_y_length] #####
    min_y_length = min(y_length)
    y_seeds_new = np.zeros((seed_count, min_y_length))

    for i in range(seed_count):
        y_seeds_new[i] = y_seeds[i][0:min_y_length]
    
    y_seeds = y_seeds_new # [4, min_y_len]
    print(y_seeds.shape)
    
    ###### plot y_seeds with color_mean, scatter_marker, std_shadow #############
    y_mean = y_seeds.mean(axis = 0)
    x_mean = [i for i in range(min_y_length)]
    ax.plot(x_mean, y_mean, color=color, linewidth = linewidth)
    
    x_scatter = np.linspace(0, min_y_length-1, int(min_y_length/scatter_space), dtype=np.int)
    y_scatter = y_mean[x_scatter]
    ax.scatter(x_scatter, y_scatter, color=color, label=label, marker=marker, s=120)
    
    y_std = y_seeds.std(axis = 0)
    upper_bound = y_mean + y_std
    lower_bound = y_mean - y_std
    ax.fill_between(x_mean, upper_bound, lower_bound, where=upper_bound>lower_bound,
                    facecolor=color, interpolate = True, alpha = alpha)
    


def main():
    data_n_path = []
    data_5 = "../results/latest_version5/SAC/*"
    data_6 = "../results/latest_version6/SAC/*"
    data_7 = "../results/latest_version7/SAC/*"
    data_9 = "../results/latest_version9/SAC/seed3"
    data_11 = "../results/measure11/sac/measure_every_n_steps10/seed1"

    for path in [data_5, data_6, data_7, data_9,data_11]:
        data_n_path.append(glob.glob(path))
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlabel('# episode', fontsize = 33)
    ax.set_ylabel('reward', fontsize = 33)

    index = 0
    for nbit in [5, 6, 7, 9, 11]:
        print("reading {} data path".format(nbit))
        plot_func(ax, data_n_path[index], color_dictionary[index], '{}'.format(nbit), marker_dictionary[index])
        index += 1

    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(30)
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(30)

    ax.legend(loc='lower right', fontsize = 32, markerscale=0.9)

    for limit in [2000, 3000, 5000]:
        ax.set_xlim(0, limit)
        plt.savefig('SAC_11qubit{}.pdf'.format(limit), dpi = 100, bbox_inches='tight')
        plt.savefig('SAC_11qubit{}.jpg'.format(limit), dpi = 100, bbox_inches='tight')


if __name__ == '__main__':
    main()
    # data_9 = "../results/latest_version9/SAC/*"
    # paths = glob.glob(data_9)
    # print(paths[0], paths[1])
    # event = EventAccumulator(paths[0])
    # event.Reload()

    # print(event.scalars.Keys())
    # y = event.scalars.Items('episode_threshold') 
    # y_len = len(y)                              
    # print(y_len)
    