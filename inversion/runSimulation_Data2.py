import os.path
import sys
import numpy as np
import time
import datetime
import h5py
import configparser

from makeDepthScan import depth_scan_from_hdf, depth_scan_from_hdf_data, depth_scan_from_hdf_data_IR

sys.path.append('../')

import util
import paraPropPython as ppp
from receiver import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rxList_from_file, create_transmitter_array, create_hdf_FT
from data import create_tx_signal, bscan, bscan_rxList, create_hdf_bscan

if len(sys.argv) == 6:
    fname_config = sys.argv[1] #The Config File -> sys.argv[1]
    fname_data = sys.argv[2] # This must contain the date or the psuedo-data -> bscan, sys.argv[2]
    fname_n_matrix = sys.argv[3] # I use this to store the results AND the simulation parameters sys.argv[3]
    ii_generation = int(sys.argv[4]) #The Generation Number of the n_profile sys.argv[4]
    jj_select = int(sys.argv[5]) #The individual number from that Generation sys.argv[5]
    fname_out = None
elif len(sys.argv) == 7:
    fname_config = sys.argv[1]  # The Config File -> sys.argv[1]
    fname_data = sys.argv[2]  # This must contain the date or the psuedo-data -> bscan, sys.argv[2]
    fname_n_matrix = sys.argv[3]  # I use this to store the results AND the simulation parameters sys.argv[3]
    ii_generation = int(sys.argv[4])  # The Generation Number of the n_profile sys.argv[4]
    jj_select = int(sys.argv[5])  # The individual number from that Generation sys.argv[5]
    fname_out = sys.argv[6]
else:
    print('incorrect arg number')
    sys.exit()
#=======================

depth_scan_from_hdf_data_IR(fname_config=fname_config, fname_n_matrix=fname_n_matrix,
                    ii_generation=ii_generation, jj_select=jj_select,
                    fname_data = fname_data, fname_out=fname_out)