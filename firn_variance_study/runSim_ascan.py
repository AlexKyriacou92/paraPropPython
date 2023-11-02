import os.path
import sys
import numpy as np
import time
import datetime
import h5py
import configparser

sys.path.append('../')


import util
import paraPropPython as ppp
from receiver import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rxList_from_file, create_transmitter_array, create_hdf_FT
from data import create_tx_signal, bscan, bscan_rxList, create_hdf_bscan
from makeDepthScan import depth_scan_impulse_smooth, run_field

from ascan_multi import run_scan_impulse

if len(sys.argv) == 7:
    fname_config = sys.argv[1] #The Config File -> sys.argv[1]
    fname_output = sys.argv[2]
    fname_nprof = sys.argv[3]

    ii_year = int(sys.argv[4])
    ii_freq = int(sys.argv[5])
    ii_tx = int(sys.argv[6])
else:
    sys.exit()

nprofile_hdf = h5py.File(fname_nprof, 'r')
nprof_mat = np.array(nprofile_hdf.get('n_profile_matrix'))
zprof_mat = np.array(nprofile_hdf.get('z_profile'))
sim0 = create_sim(fname_config)
dz = sim0.dz
zprof_data = np.arange(min(zprof_mat), sim0.iceDepth, dz)
nprof_data = np.interp(zprof_data, zprof_mat, nprof_mat[ii_year]).real

if np.any(np.isnan(nprof_data) == True) == True:
    print('error, undefined values of ref index array!')
    sys.exit(-1)

run_scan_impulse(fname_config,
                 fname_output,
                 nprof_data,
                 zprof_data,
                 ii_freq,
                 ii_tx)