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
    0:"o", 1:"^", 2:"D", 3:"x"
}
linestyle_dictionary = {
    0:"-", 1:"--", 2:":", 3:"-."
}


def plot_func(ax, method_dirs, color, label, marker, smooth_index=15, alpha=0.4, linewidth=2.0, scatter_space = 400):
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
        if (label == 'sac'):
            y = event.scalars.Items('episode_threshold') #  threshold_value
        else:
            y = event.scalars.Items('episode_reward')
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
    ax.scatter(x_scatter, y_scatter, color=color, label=label, marker=marker,s = 100)
    
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

def main_plot():
    # figure & axes
    fig, axs = plt.subplots(1, 2, figsize=(24, 8))
    #fig.suptitle('Nqubit', fontsize = 20)

    # config respective axes
    for i in range(2):
        axs[i].set_xlabel('# episode', fontsize = 35)
        axs[i].set_ylabel('reward', fontsize = 35)
        #axs[i].set_title('{0}-bits'.format(i + 5), fontsize = 20)

    print('start plotting')
    # plot axes
    ax_count = 0
    for nbit in [5, 7]:                                   # key_part to change [5, 6, 7]
        # td3,sac,ddpg dirs
        print('plotting {0} axes'.format(ax_count + 1))
        data_path_n = '../results/latest_version{0}/*'.format(nbit)
        final_path, method_names = dir_process(data_path_n)

        '''
        # dqn_dirs
        dqn_data_path = '../results/DQN_data/{0}/'.format(nbit)
        path_to_txt = get_txt_path(dqn_data_path)
        
        final_path.append(path_to_txt[0:4])
        method_names.append('dqn')
        '''

        # plot 4 curve for 3 axes
        for i in range(len(method_names)):
            plot_func(axs[ax_count], final_path[i], color_dictionary[i], method_names[i], marker_dictionary[i])

        ax_count += 1
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, shadow=True, ncol=3, fontsize = 25,markerscale=1.2)

    ###
    limits = [8000, 10000]
    plt.tight_layout()

    print('saving figures')
    for limit in limits:
        for i in range(2):
            axs[i].set_xlim(0, limit)
            for label in axs[i].xaxis.get_ticklabels():
                label.set_fontsize(30)
            for label in axs[i].yaxis.get_ticklabels():
                label.set_fontsize(30)
            
        plt.savefig('{}episodes.pdf'.format(limit), dpi = 100, bbox_inches='tight')
        plt.savefig('{}episodes.jpg'.format(limit), dpi = 100, bbox_inches='tight')
        
        



if __name__ == '__main__':
    main_plot()
    