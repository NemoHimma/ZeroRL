# nbit-6, with truncation, one figure, all T.
# with smooth

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

smooth = True
smooth_index = 150 # [0, max)
threshold_len = 0 # == 10000??????????
linewidth = 1.0

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
        plt.plot([i.step for i in threshold], tmp_threshold, color=color, label=label, linewidth=linewidth) 
    else:
        event = EventAccumulator(dir)
        event.Reload()
        threshold = event.scalars.Items('threshold')
        plt.plot([i.step for i in threshold], [i.value for i in threshold], color=color, label=label, linewidth=linewidth)
        return

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


dir2 = "/home/pami/ZeroRL/nqubit/results/sac_energy_new/nbit-5/T-1.450seed-6/events.out.tfevents.1608019741.pami12"
plot_func(dir=dir2, color='steelblue', label='T=3.5')

dir3 = "/home/pami/ZeroRL/nqubit/results/sac_energy_new/nbit-6/T-4.000/events.out.tfevents.1607686837.pami12"
plot_func(dir=dir3, color='g', label='T=4.0')

dir4 = "/home/pami/ZeroRL/nqubit/results/sac_energy_new/nbit-6/T-4.500/events.out.tfevents.1607686906.pami12"
plot_func(dir=dir4, color='gold', label='T=4.5')

dir5 = "/home/pami/ZeroRL/nqubit/results/sac_energy_new/nbit-6/T-5.000/events.out.tfevents.1607676571.pami12"
plot_func(dir=dir5, color='grey', label='T=5.0')

dir6 = "/home/pami/ZeroRL/nqubit/results/sac_energy_new/nbit-6/T-7.000/events.out.tfevents.1607676668.pami12"
plot_func(dir=dir6, color='sandybrown', label='T=7.0')

dir7 = "/home/pami/ZeroRL/nqubit/results/sac_energy_new/nbit-6/T-9.000/events.out.tfevents.1607676748.pami12"
plot_func(dir=dir7, color='pink', label='T=9.0')

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





