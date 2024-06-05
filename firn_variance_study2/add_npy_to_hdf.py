import os.path
import sys
import numpy as np
import time
import datetime
import h5py
import configparser
from numpy import exp, log
import matplotlib.pyplot as pl
from makeDepthScan import depth_scan_impulse_smooth
from ku_scripting import *

sys.path.append('../')
from paraPropPython import paraProp as ppp
import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rxList_from_file, create_tx_signal, create_transmitter_array
from data import create_hdf_bscan, bscan_rxList, create_hdf_FT
from data import create_ascan_hdf, ascan

import util
from data import create_transmitter_array_from_file

path2sim = sys.argv[1]

hdf_list = []
npy_list = []
path_list = os.listdir(path2sim)
for file in path_list:
    if file.endswith('.h5'):
        hdf_list.append(file)
        file_npy = file[:-3] + '.npy'
        npy_list.append(file_npy)

nFiles = len(hdf_list)

for i in range(nFiles):
    fname_hdf = hdf_list[i]
    fname_npy = npy_list[i]
    ascan_i = ascan()
    ascan_i.load_from_hdf(fname_hdf=fname_hdf)
    ascan_i.save_spectrum(fname_npy=fname_npy)