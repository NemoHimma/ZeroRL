import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator



color_dictionary = {
    0: "#1F618D", 1: "#F322CD", 2: "#0E0F0F", 3: "#7FB3D5", 4: "#22DAF3",
    5: "#5B2C6F", 6: "#800000", 7: "#008000", 8: "#008000", 9: "#E74C3C",
    10: "#D35400", 11: "#800000", 12: "#2980B9", 13: "#F1948A", 14: "#1C2833",
    15: "#E74C3C", 16: "#0000FF" 
}

marker_dictionary = {
    0:"o", 1:"^", 2:"D", 3:"x", 4:">", 5:"1", 6:"p", 7:"P", 8:"*"
}

def plot_func(ax, method_dirs, color, label, marker, smooth_index=30, alpha=0.5, linewidth=2.0, scatter_space = 400):
    '''
    input: method_dirs : ['algo1/seed1','~algo1/seed4']
    '''
    ##### extrat data from all seeds #####
    y_seeds = []
    y_length = []
    seed_count = len(method_dirs)
    
    for dir in method_dirs:
        event = EventAccumulator(dir)
        event.Reload()
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
    

def dir_process(data_path):
    '''
    data_path = './results/latest_version5/*'
    final_path = '[['algo1/seed1','~algo1/seed4'], ['~/algo2/seed1', '~/algo2/seed4']]'
    '''

    method_paths = glob.glob(data_path)

    final_path = []
    for i in range(len(method_paths)):
        final_path.append(glob.glob(method_paths[i] + "/*"))
    
    method_names = []
    for path in method_paths:
        method_names.append(os.path.basename(path))

    return final_path, method_names

if __name__ == '__main__':
    #data_path = '../results/latest_episode_length5/sac/*'
    data_path = '../results/latest_reward_scale5/sac/*'
    final_path, method_names = dir_process(data_path)
    final_path, method_names = sorted(final_path), sorted(method_names)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlabel('# episode', fontsize = 25)
    ax.set_ylabel('reward', fontsize = 25)

    for i in range(len(method_names)):
        print('reading {} episode_length'.format(i+1))
        plot_func(ax, final_path[i], color_dictionary[i], method_names[i], marker_dictionary[i])

    print('saving figures')
    

    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(20)
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(20)

    ax.legend(loc='lower right', fontsize = 20, markerscale=0.9)
    plt.savefig('reward_scale_comparision.pdf', dpi = 100, bbox_inches='tight')