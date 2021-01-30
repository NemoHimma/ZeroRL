import numpy as np
import glob
import os
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

def dir_process(data_path):
    '''
    data_path = '../results/latest_reward_scale5/sac/*'
    '''

    all_paths = glob.glob(data_path)
    legend_names = []
    for path in all_paths:
        legend_names.append(os.path.basename(path))
    return all_paths, legend_names


def plot_reward_scale(ax, data_path_n, n, linewidth = 2.0, scatter_space = 1000, smooth_index=30):
    '''
    
    '''
    ## paths_to_all_files & legend_names to plot
    all_paths, legend_names = dir_process(data_path_n)

    num_curves = len(all_paths)
    print(num_curves)
    ax.set_xlabel('#step', fontsize = 20)
    ax.set_ylabel('reward', fontsize = 20)
    ax.set_title('{0}-bit'.format(n), fontsize = 20)

    # read_data
    data = []
    data_length = []

    for directory in all_paths:
        event = EventAccumulator(directory)
        event.Reload()
        y = event.scalars.Items('episode_threshold') #  threshold_value
        y_len = len(y)                               # threshold_len
        data_length.append(y_len)
        print("reading")

        ## smoothing
        smooth_array = np.zeros(y_len)
        for i in range(y_len):
            smooth_array[i] = np.array([j.value for j in y[i:i+smooth_index]]).mean()
        data.append(smooth_array)
        
    print("finished reading")
    # process_data
    min_length = min(data_length)
    data_reshape = np.zeros((num_curves, min_length))

    for i in range(num_curves):
        data_reshape[i] = data[i][0:min_length]
    
    data = data_reshape
    index = [i for i in range(min_length)]
    #label = ['{}-run'.format(i+1) for i in range(num_curves)]

    print("plotting")
    for num in range(num_curves):
        ax.plot(index, data[num][:], color=color_dictionary[num], linewidth= linewidth)

        x_scatter = np.linspace(0, min_length-1, int(min_length/scatter_space), dtype=np.int)
        y_scatter = data[num][x_scatter]
        ax.scatter(x_scatter, y_scatter, color=color_dictionary[num], label=legend_names[num], marker=marker_dictionary[num])

    
    ax.legend(loc='lower right', fancybox=True, shadow=True, ncol=num_curves, fontsize = 20)
    #ax.legend(loc='lower right', bbox_to_anchor=(0.5, -0.15),fancybox=True, shadow=True, ncol=num_curves, fontsize = 20)


    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(16)
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(16)

    print("saving")
    plt.savefig('{}-episode_length.jpg'.format(n), dpi = 100, bbox_inches='tight')

if __name__ == '__main__':
    fig, ax = plt.subplots(1, 1, figsize=(25, 16))
   # data_path = '../results/latest_reward_sacle5/sac/*'
    data_path = '../results/latest_episode_length5/sac/*'
    plot_reward_scale(ax, data_path, 5)

    
    