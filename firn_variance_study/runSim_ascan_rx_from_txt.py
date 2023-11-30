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
from data import create_sim, create_rxList_from_file, create_transmitter_array, create_hdf_FT, ascan
from data import create_tx_signal, bscan, bscan_rxList, create_hdf_bscan, create_transmitter_array_from_file
from makeDepthScan import depth_scan_impulse_smooth, run_field, run_ascan_rx, run_ascan_rx_txt

if len(sys.argv) == 7:
    fname_config = sys.argv[1] #The Config File -> sys.argv[1]
    fname_spectrum = sys.argv[2]
    fname_hdf = sys.argv[3]
    fname_nprof = sys.argv[4]

    ii_freq = int(sys.argv[5])
    ii_tx = int(sys.argv[6])
else:
    print('error')
    sys.exit()


nprof_data, zprof_data = util.get_profile_from_file(fname_nprof)

txList = create_transmitter_array_from_file(fname_config)
z_tx = txList[ii_tx]

ascan_in = ascan()
ascan_in.load_from_hdf(fname_hdf=fname_hdf)
tx_signal_in = ascan_in.tx_signal
tx_spectrum = tx_signal_in.get_spectrum()
freq_space = tx_signal_in.get_freq_space()
freq = freq_space[ii_freq]

if np.any(np.isnan(nprof_data) == True) == True:
    print('error, undefined values of ref index array!')
    sys.exit(-1)

suffix = fname_spectrum[-3:]

if suffix == 'npy':
    run_ascan_rx(fname_config=fname_config, n_profile=nprof_data, z_profile=zprof_data,
             z_tx=z_tx, freq=freq, fname_hdf=fname_hdf, fname_npy=fname_spectrum)
elif suffix == 'txt':
    run_ascan_rx_txt(fname_config=fname_config, n_profile=nprof_data, z_profile=zprof_data,
             z_tx=z_tx, freq=freq, fname_hdf=fname_hdf, fname_txt=fname_spectrum)
else:
    print('Wrong file ending', suffix, ' for ', fname_spectrum, ', you have to use npy or txt')
    sys.exit()
