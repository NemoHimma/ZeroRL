# nbit-6, with truncation, one figure, all T.
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_func(dir, color, label):
    event = EventAccumulator(dir)
    event.Reload()
    threshold = event.scalars.Items('threshold')
    plt.plot([i.step for i in threshold], [i.value for i in threshold], color=color, label=label, linewidth=0.5)
    return

dir = "/home/pami/ZeroRL/nqubit/results/sac_energy_new/nbit-6/T-3.000/events.out.tfevents.1607676640.pami12"
event = EventAccumulator(dir)
event.Reload()
print("\n\n")
print("event_acc: ", event)
print("ea.scalars.Keys(): ", event.scalars.Keys())

threshold = event.scalars.Items('threshold')
print("len(threshold): ", len(threshold))
print("threshold[0:2]: ", threshold[0:2])
print([(i.step, i.value) for i in threshold[0:11]])


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

plt.plot([i.step for i in threshold], [i.value for i in threshold], color='maroon', label='T=3.0')
plt.xlim(0)
# acc=ea.scalars.Items('acc')
# ax1.plot([i.step for i in acc],[i.value for i in acc],label='acc')
plt.xlabel("step")
plt.ylabel("threshold")

dir2 = "/home/pami/ZeroRL/nqubit/results/sac_energy_new/nbit-6/T-3.500/events.out.tfevents.1607686863.pami12"
event2 = EventAccumulator(dir2)
event2.Reload()
threshold2 = event2.scalars.Items('threshold')
plt.plot([i.step for i in threshold2], [i.value for i in threshold2], color='steelblue', label='T=3.5')

dir3 = "/home/pami/ZeroRL/nqubit/results/sac_energy_new/nbit-6/T-4.000/events.out.tfevents.1607686837.pami12"
event3 = EventAccumulator(dir3)
event3.Reload()
threshold3 = event3.scalars.Items('threshold')
plt.plot([i.step for i in threshold3], [i.value for i in threshold3], color='g', label='T=4.0')

dir4 = "/home/pami/ZeroRL/nqubit/results/sac_energy_new/nbit-6/T-4.500/events.out.tfevents.1607686906.pami12"
plot_func(dir=dir4, color='gold', label='T=4.5')

dir5 = "/home/pami/ZeroRL/nqubit/results/sac_energy_new/nbit-6/T-5.000/events.out.tfevents.1607676571.pami12"
plot_func(dir=dir5, color='grey', label='T=5.0')

dir6 = "/home/pami/ZeroRL/nqubit/results/sac_energy_new/nbit-6/T-7.000/events.out.tfevents.1607676668.pami12"
plot_func(dir=dir6, color='sandybrown', label='T=7.0')

dir7 = "/home/pami/ZeroRL/nqubit/results/sac_energy_new/nbit-6/T-9.000/events.out.tfevents.1607676748.pami12"
plot_func(dir=dir7, color='pink', label='T=9.0')

plt.legend(loc='lower right')
plt.savefig('original.jpg')

plt.xlim(0, 15000)
plt.savefig('truncation_x.jpg')

plt.ylim(-2)
plt.savefig('truncation_x_y.jpg')

plt.xlim(0, 3000)
plt.savefig('truncation_3000_-2.jpg')

plt.xlim(0, 1000)
plt.savefig('truncation_1000_-2.jpg')





