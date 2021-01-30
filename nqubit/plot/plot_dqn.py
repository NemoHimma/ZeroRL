import numpy as np
import glob
import os
import matplotlib.pyplot as plt

color_dictionary = {
    0: "#1F618D", 1: "#F322CD", 2: "#0E0F0F", 3: "#7FB3D5", 4: "#22DAF3",
    5: "#5B2C6F", 6: "#800000", 7: "#008000", 8: "#008000", 9: "#E74C3C",
    10: "#D35400", 11: "#800000", 12: "#2980B9", 13: "#F1948A", 14: "#1C2833",
    15: "#E74C3C", 16: "#0000FF" 
}

marker_dictionary = {
    0:"o", 1:"^", 2:"D", 3:"x", 4:">", 5:"1", 6:"p", 7:"P", 8:"*"
}
def get_txt_path(dqn_path_n):

    '''

    input: dqn_path_n = './results/DQN_data/5/'
    output: ['./results/DQN_data/5/eachstep50.226368.txt', './results/DQN_data/5/eachstep50.232696.txt']

    '''

    data_path = glob.glob(dqn_path_n + '*')
    base_names = []

    for txt in data_path:
        base_names.append(os.path.basename(txt))
    each_step_txt = []

    for txt in base_names:
        if txt.startswith('eachstep'):
            each_step_txt.append(txt)

    path_to_txt = [dqn_path_n + txt for txt in each_step_txt]
    return path_to_txt

def read_data_from_txt(path_to_single_txt):
    '''
    input: ./results/DQN_data/5/eachstep50.226368.txt
    '''
    str_data = []
    line_count = 1
    with open(path_to_single_txt, 'r') as f:
        for line in f:
            if line.startswith('-'):
                str_data.append(line)
                line_count += 1
            else:
                str_data[line_count - 2] = str_data[line_count - 2] + line
    
    step_threshold = []
    for i in range(len(str_data)):
        number = np.float(str_data[i].split('\n')[0].split('_')[0])
        step_threshold.append(number)

    return step_threshold

def plot_n_bit_dqn(ax, dqn_path_n, n, linewidth = 3.0, scatter_space = 1000, smooth_index=60):
    '''
    input: dqn_path_n = '../results/DQN_data/5/'
    '''
    path_to_txt = get_txt_path(dqn_path_n)

    num_curves = len(path_to_txt)
    if num_curves > 4:
        num_curves = 4

    ax.set_xlabel('#step', fontsize = 30)
    ax.set_ylabel('reward', fontsize = 30)
  #ax.set_title('{0}-bit'.format(n), fontsize = 20)

    # read_data
    data = []
    data_length = []

    for i in range(num_curves):
        step_threshold = read_data_from_txt(path_to_txt[i])
        step_len = len(step_threshold)
        ##### smoothing #####
        smooth_array = np.zeros(step_len)
        for ele in range(step_len):
            smooth_array[ele] = np.array([j for j in step_threshold[ele:ele+smooth_index]]).mean()

        data.append(smooth_array)
        data_length.append(step_len)
        

    min_length = min(data_length)
    data_reshape = np.zeros((num_curves, min_length))

    for i in range(num_curves):
        data_reshape[i] = data[i][0:min_length]
    
    data = data_reshape
    index = [i for i in range(min_length)]
    label = ['{}-run'.format(i+1) for i in range(num_curves)]

    for num in range(num_curves):
        ax.plot(index, data[num][:], color=color_dictionary[num], linewidth= linewidth, markersize=12)

        x_scatter = np.linspace(0, min_length-1, int(min_length/scatter_space), dtype=np.int)
        y_scatter = data[num][x_scatter]
        ax.scatter(x_scatter, y_scatter, color=color_dictionary[num], label=label[num], marker=marker_dictionary[num])

    
    ax.legend(loc='lower right', fontsize = 20,markerscale=2)


    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(20)
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(20)
        

    plt.savefig('{}-bit-dqn.jpg'.format(n), dpi = 100, bbox_inches='tight')
        
        



if __name__ == '__main__':
    fig, ax = plt.subplots(1, 1, figsize = (25, 16))
    #ax_count = 0
    #for nbit in [5, 6, 7, 9, 11]:
    nbit = 11
    dqn_path_n = '../results/DQN_data/{}/'.format(nbit)
    plot_n_bit_dqn(ax, dqn_path_n, nbit)
        
    
