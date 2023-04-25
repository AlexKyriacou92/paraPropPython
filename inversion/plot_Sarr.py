import os.path
import sys
import numpy as np
import time
import datetime
import h5py
import configparser
from scipy.interpolate import interp1d
from makeDepthScan import depth_scan_from_hdf
import matplotlib.pyplot as pl

sys.path.append('../')

import util
import paraPropPython as ppp
from receiver import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rxList_from_file, create_transmitter_array, create_hdf_FT
from data import create_tx_signal, bscan, bscan_rxList, create_hdf_bscan

fname_nmatrix = sys.argv[1]

nmatrix_hdf = h5py.File(fname_nmatrix, 'r')
n_profile_matrix = np.array(nmatrix_hdf['n_profile_matrix'])
S_arr = np.array(nmatrix_hdf['S_arr'])
genes_arr = np.array(nmatrix_hdf['genes_matrix'])
z_genes = np.array(nmatrix_hdf['z_genes'])
nGenes = len(z_genes)
z_profile = abs(np.array(nmatrix_hdf['z_profile']))[1:-1]
nmatrix_hdf.close()

nGenerations = len(S_arr)
nGenerations_finished = 0
S_max_list = []
S_med_list = []
S_mean_list = []
gens = []
for i in range(nGenerations):
    if np.all(S_arr[i] == 0) == False:
        S_max = max(S_arr[i])
        S_mean = np.mean(S_arr[i])
        S_median = np.median(S_arr[i])
        S_max_list.append(S_max)
        S_med_list.append(S_median)
        S_mean_list.append(S_mean)
        gens.append(i)
S_max_list = np.array(S_max_list)
S_mean_list = np.array(S_mean_list)
S_med_list = np.array(S_med_list)
fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
ax.plot(gens, S_max_list,c='b',label='Max')
ax.plot(gens, S_med_list, c='g',label='Median')
ax.plot(gens, S_mean_list,c='r',label='Mean')
ax.set_xlabel('Generation')
ax.legend()
ax.set_ylabel('Fitness Score $S = (\sum_{ij }m_{ij})^{-1} $')
ax.grid()
fig.savefig('S_score.png')
pl.show()

fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
ax.plot(gens, 1/S_max_list,c='b',label='Max')
ax.plot(gens, 1/S_med_list, c='g',label='Median')
ax.plot(gens, 1/S_mean_list,c='r',label='Mean')
ax.set_xlabel('Generation')
ax.legend()
ax.set_ylabel('Global Misfit Score $\sum_{ij} m_{ij}$')
ax.grid()
fig.savefig('M_score.png')
pl.show()