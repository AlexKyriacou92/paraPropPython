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
from makeDepthScan import depth_scan_impulse_smooth

if len(sys.argv) == 4:
    fname_config = sys.argv[1] #The Config File -> sys.argv[1]

    fname_nprof = sys.argv[2]
    fname_nprof_split = fname_nprof.split('.')
    suffix = fname_nprof_split[-1]
    if suffix == 'h5' or suffix == 'hdf5'
        ii_select = int(sys.argv[3])
    elif suffix == 'txt':
        fname_out = sys.argv[3]
    else:
        print('Error! Ref-profile file fname_nprof has invalid suffix: ', suffix, ', suffix must be format .h5 or .hdf5 or .txt')
        sys.exit()

    fname_out = None
elif len(sys.argv) == 5:
    fname_config = sys.argv[1]  # The Config File -> sys.argv[1]

    fname_nprof = sys.argv[2]
    fname_nprof_split = fname_nprof.split('.')
    suffix = fname_nprof_split[-1]

    ii_select = int(sys.argv[3])

    fname_out = sys.argv[4]
else:
    print('incorrect arg number')
    sys.exit()
#=======================
if suffix == 'h5' or suffix == 'hdf5':
    nprofile_hdf = h5py.File(fname_nprof, 'r')
    nprof_mat = np.array(nprofile_hdf.get('n_profile_matrix'))
    zprof_mat = np.array(nprofile_hdf.get('z_profile'))
    sim0 = create_sim(fname_config)
    dz = sim0.dz
    zprof_data = np.arange(min(zprof_mat), sim0.iceDepth, dz)
    nprof_data = np.interp(zprof_data, zprof_mat, nprof_mat[ii_select]).real
elif suffix == 'txt':
    nprof_data, zprof_data = util.get_profile_from_file(fname_nprof)

if np.any(np.isnan(nprof_data) == True) == True:
    print('error, undefined values of ref index array!')
    sys.exit(-1)

depth_scan_impulse_smooth(fname_config=fname_config, n_profile=nprof_data, z_profile=zprof_data, fname_out=fname_out)