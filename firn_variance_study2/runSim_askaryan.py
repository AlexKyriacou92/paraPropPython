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
from makeDepthScan import depth_scan_askaryan

if len(sys.argv) == 4:
    fname_config = sys.argv[1] #The Config File -> sys.argv[1]
    fname_nprof = sys.argv[2]
    fname_out = sys.argv[3]
else:
    print('Run arg number, abort')
    sys.exit()

nprof_data, zprof_data = util.get_profile_from_file(fname_nprof)
bscan_npy = depth_scan_askaryan(fname_config=fname_config,
                                n_profile=nprof_data,
                                z_profile=zprof_data,
                                fname_out=fname_out)