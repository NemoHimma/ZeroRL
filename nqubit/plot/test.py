import numpy as np
import glob
import os
import re

appointed_folder = "nbit-5"

a = np.array([1, 2, 3, 4, 5, 6])
b = np.zeros(10)
print(b)
print(a.mean())
print(os.path)

original_path = "../results/sac_energy_new/" + appointed_folder + "/*"
print(original_path)
paths = glob.glob(original_path) # type:string
print("paths: ", paths)
