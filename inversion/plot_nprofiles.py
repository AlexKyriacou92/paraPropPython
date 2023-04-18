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

nArgs = len(sys.argv)
nArgs_required = 4
if nArgs == nArgs_required:
    fname_nmatrix = sys.argv[1]
    fname_pseudo = sys.argv[2]
    path2plots = sys.argv[3]
else:
    print('nArgs =', nArgs, 'is wrong, should be:', nArgs_required)
    print('Please enter: python ',  sys.argv[0], ' <path/to/fname_nmatrix> <path/to/plot_dir>')
    sys.exit()

nmatrix_hdf = h5py.File(fname_nmatrix, 'r')
n_profile_matrix = np.array(nmatrix_hdf['n_profile_matrix'])
S_arr = np.array(nmatrix_hdf['S_arr'])
z_profile = np.array(nmatrix_hdf['z_profile'])
nmatrix_hdf.close()

nGenerations = len(S_arr)
nIndividuals = len(S_arr[0])
nDepths = len(z_profile)

nGenerations_complete = 0
for i in range(nGenerations):
    S_arr_gen = S_arr[i]
    if np.all(S_arr_gen == 0) == False:
        nGenerations_complete += 1

bscan_pseudo = bscan_rxList()
bscan_pseudo.load_sim(fname_pseudo)
z_profile_pseudo = bscan_pseudo.z_profile
n_profile_pseudo = bscan_pseudo.n_profile

f_pseudo_interp = interp1d(z_profile_pseudo, n_profile_pseudo)
n_profile_pseudo_interp = f_pseudo_interp(z_profile)

if os.path.isdir(path2plots) == False:
    os.system('mkdir ' + path2plots)

for i in range(nGenerations_complete):
    S_max = max(S_arr[i])
    ii_max = np.argmax(S_arr)
    n_profile_max = n_profile_matrix[i, ii_max]
    n_residuals_max = n_profile_max - n_profile_pseudo_interp

    fig = pl.figure(figsize=(10,8), dpi=100)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(n_profile_max, z_profile,c='b', label='Best Score')
    ax1.plot(n_profile_pseudo_interp, z_profile,c='k', label='Truth')
    ax1.grid()
    ax1.set_xlabel('Ref Index n')
    ax1.set_ylabel('Depth z [m]')
    ax1.set_ylim(16,0)
    ax1.set_xlim(1.0, 2.0)
    ax1.legend()

    ax2.plot(n_residuals_max, z_profile, label='Residuals, Best Score')
    ax2.grid()
    ax2.set_xlabel('Ref Index Residuals $\Delta n$')
    ax2.legend()

    fname_plot = path2plots + '/' + 'ref_index_gen' + str(ii_max).zfill(3) + '_plot.png'
    fig.savefig(fname_plot)
    pl.close(fig)
