import numpy as np
import math
import h5py
import sys

fname_out = sys.argv[1]


rx_ranges = np.arange(50, 1000, 50)
rx_depths = np.arange(10, 300, 20)
nRanges = len(rx_ranges)
nDepths = len(rx_depths)

with open(fname_out, 'w') as fout:
    for i in range(nRanges):
        for j in range(nDepths):
            x_ij = rx_ranges[i]
            z_ij = rx_depths[j]
            line = str(round(x_ij, 3)) + '\t' + str(round(z_ij, 3)) + '\n'
            fout.write(line)
