import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator



color_dictionary = {
    0: "#1F618D", 1: "#F322CD", 2: "#D35400", 3: "#7FB3D5", 4: "#F1948A",
    5: "#5B2C6F", 6: "#800000", 7: "#008000", 8: "#008000", 9: "#E74C3C",
    10: "#0E0F0F", 11: "#800000", 12: "#2980B9", 13: "#22DAF3", 14: "#1C2833",
    15: "#E74C3C", 16: "#0000FF" 
}

marker_dictionary = {
    0:"o", 1:"^", 2:"D", 3:"x", 4:">", 5:"1", 6:"p", 7:"P", 8:"*"
}

def plot_func(ax, method_dirs, color, label, marker, smooth_index=10, alpha=0.45, linewidth=1.2, scatter_space = 400):
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

def order_reward_scale(final_path, method_names):

    final_path_order, method_names_order = [], []
    index = [1, 3, 2, 0]
    for i in index:
        final_path_order.append(final_path[i])
        method_names_order.append(method_names[i])

    return final_path_order, method_names_order
   




if __name__ == '__main__':
    #data_path = '../results/latest_episode_length5/sac/*'
    #data_path = '../results/latest_reward_scale5/sac/*'
    #data_path = '../results/measure5/sac/*'
    data_path  = '../results/EnvSetting5/*'
    #data_path = '../results/latest_version7/*'
    final_path, method_names = dir_process(data_path)
    print(final_path, method_names)

    final_path, method_names = order_reward_scale(final_path, method_names)
    # method_names[0] = r'$\Delta t_{\rm{w}}=2$'
    # method_names[1] = r'$\Delta t_{\rm{w}}=5$'
    # method_names[2] = r'$\Delta t_{\rm{w}}=10$'
    # method_names[3] = r'$\Delta t_{\rm{w}}=15$'
    #final_path, method_names = sorted(final_path), sorted(method_names)
    print(final_path, method_names)
    
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlabel('# episode', fontsize = 33)
    ax.set_ylabel('reward', fontsize = 33)

    for i in range(len(method_names)):
        print('reading {} setting'.format(i+1))
        plot_func(ax, final_path[i], color_dictionary[i], method_names[i], marker_dictionary[i])

    print('saving figures')
    

    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(30)
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(30)

    ax.legend(loc='lower right', fontsize = 31, markerscale=0.9)

    #ax.set_xlim(0, 10000)
    #plt.savefig('y_tune_setting.pdf', dpi = 100, bbox_inches='tight')
    ax.set_ylim(-2.5, -0.95)
    ax.set_xlim(0, 5000)
    plt.savefig('setting1.pdf', dpi = 100, bbox_inches='tight')
    plt.savefig('setting1.jpg', dpi = 100, bbox_inches='tight')