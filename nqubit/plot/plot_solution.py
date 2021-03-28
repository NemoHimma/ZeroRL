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

def order_reward_scale(final_path):

    final_path_order = []
    index = [1, 5, 0, 2, 3, 4]
    for i in index:
        final_path_order.append(final_path[i])
        

    return final_path_order

def get_txt_path(data_path):
    '''
     data_path = "../results/solution/*"
    '''
    final_path = order_reward_scale(glob.glob(data_path))
    return final_path

def read_data_from_txt(path_to_solution, length):
    y_solution = []
    y_length = []
    
    for solution_path in path_to_solution:
        print("reading")
        event = EventAccumulator(solution_path)
        event.Reload()
        print(event.scalars.Keys())
        y = event.scalars.Items('soluiton') 
        y_len = len(y)
        y_length.append(y_len)
        y_solution.append(y)
        

    print(y_length)
    min_y_length = min(y_length)
    if length > min_y_length:
        length = min_y_length

    reshape_y = np.zeros((6, min_y_length))
    for curve in range(6):
        tmp = []
        for step in range(length):
            tmp.append(y_solution[curve][step].value)
        reshape_y[curve] = np.array(tmp)

    return reshape_y, length
    

if __name__ == '__main__':
    data_path = "../results/solution2/*"

    # data_path, legend_names
    final_path = get_txt_path(data_path)
    legend_names = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6']

    # predefined
    preset_length = 100000
    scatter_space = 500
    linewidth = 0.7
   
    data, modified_length = read_data_from_txt(final_path, preset_length)
    y_new = data
    print(y_new.shape)
    
    
     # line & scatter
    y_line = y_new
    y_scatter = y_new
    print("y_line,y_scatter:shape{}".format(y_line.shape))
    
    x_line = [i for i in range(modified_length)]
    x_scatter = np.linspace(0, modified_length-1, int(modified_length/scatter_space), dtype=np.int)

    ###### Building Graph ######
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    #ax.set_xlabel('# episode', fontsize = 33)
    #ax.set_ylabel('value', fontsize = 33)

    for i in range(len(legend_names)):
        print('reading {} solution'.format(i+1))
        ax.plot(x_line, y_line[i][0:modified_length], color=color_dictionary[i], linewidth=linewidth)
        ax.scatter(x_scatter, y_scatter[i][x_scatter], color=color_dictionary[i], label=legend_names[i], marker=marker_dictionary[i], s=150)

    print('saving figures')
    
    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(30)
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(30)

    #ax.legend(loc='lower right', fontsize = 32, markerscale=0.9)
    ax.legend(loc='upper center',bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=6, fontsize = 32, markerscale=1.0)

    #ax.set_xlim(0, 10000)
    #plt.savefig('y_tune_setting.pdf', dpi = 100, bbox_inches='tight')
    #ax.set_ylim(-2.4, -0.9)
    #ax.set_xlim(0, 5000)
    plt.savefig('solution.pdf', dpi = 100, bbox_inches='tight')
    plt.savefig('solutiong.jpg', dpi = 100, bbox_inches='tight')
