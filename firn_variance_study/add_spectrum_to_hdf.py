import os.path
import sys
import numpy as np
import time
import h5py
import util

sys.path.append('../')
from data import ascan

fname_list = sys.argv[1]

fin_list = open(fname_list, 'r')
cols0 = fin_list.readline().split()
dir_sim_path = cols0[0]
fname_hdf0 = cols0[1]
fname_npy0 = cols0[1]

fname_hdf = dir_sim_path + fname_hdf0
fname_npy = dir_sim_path + fname_npy0

cols1 = fin_list.readline().split()
nTx = int(cols1[0])
nRx = int(cols1[1])
nSamples = int(cols1[2])

cols2 = fin_list.readline()

spectrum = util.create_memmap(fname_npy, dimensions=(nTx, nRx, nSamples), data_type='complex')

for line in fin_list:
    cols_l = line.split()
    ii_freq = int(cols_l[0])
    freq_ii = float(cols_l[1])
    fname_npy_ii = dir_sim_path + cols_l[2]

    spectrum_i = np.load(fname_npy_ii, 'r')
    spectrum[:,:,ii_freq] = spectrum_i

ascan_in = ascan()
ascan_in.load_from_hdf(fname_hdf)
ascan_in.save_spectrum(fname_npy)
