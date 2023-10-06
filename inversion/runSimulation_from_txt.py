import os.path
import sys
import numpy as np
import time
import datetime
import h5py
import configparser

from makeDepthScan import depth_scan_from_hdf, depth_scan_from_txt

sys.path.append('../')

import util
import paraPropPython as ppp
from receiver import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rxList_from_file, create_transmitter_array, create_hdf_FT
from data import create_tx_signal, bscan, bscan_rxList, create_hdf_bscan

if len(sys.argv) == 3:
    fname_config = sys.argv[1] #The Config File -> sys.argv[1]
    fname_txt = sys.argv[2]
    fname_out = None
elif len(sys.argv) == 4:
    fname_config = sys.argv[1]  # The Config File -> sys.argv[1]
    fname_txt = sys.argv[2]
    fname_out = sys.argv[3]
else:
    print('incorrect arg number')
    sys.exit()
#=======================

depth_scan_from_txt(fname_config=fname_config, fname_nprofile=fname_txt, fname_out=fname_out)