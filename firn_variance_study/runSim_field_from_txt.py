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


if len(sys.argv) == 6:
    fname_config = sys.argv[1] #The Config File -> sys.argv[1]
    fname_nprof = sys.argv[2]
    z_tx = float(sys.argv[3])
    freq = float(sys.argv[4])
    fname_out = sys.argv[5]
else:
    print('Run arg number, abort')
    sys.exit()

nprof_data, zprof_data = util.get_profile_from_file(fname_nprof)

run_field(fname_config=fname_config,
          n_profile=nprof_data,
          z_profile=zprof_data,
          z_tx=z_tx,
          freq=freq,
          fname_out=fname_out)