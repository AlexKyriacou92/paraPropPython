import os.path
import sys
import numpy as np
import time
import datetime
import h5py
import configparser
from matplotlib import pyplot as pl
from makeDepthScan import depth_scan_from_hdf

sys.path.append('../')

import util
import paraPropPython as ppp
from receiver import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rxList_from_file, create_transmitter_array, create_hdf_FT
from data import create_tx_signal, bscan, bscan_rxList, create_hdf_bscan
from plotting_functions import compare_ascans
from objective_functions import misfit_function_ij

path2data = sys.argv[1]
path2sim = sys.argv[2]
z_tx = float(sys.argv[3])
x_rx = float(sys.argv[4])
z_rx = float(sys.argv[5])

bscan_data = bscan_rxList()
bscan_data.load_sim(fname=path2data)
bscan_sim = bscan_rxList()
bscan_sim.load_sim(fname=path2sim)

fname_save = 'plots/'
compare_ascans(bscan_data=bscan_data, bscan_sim=bscan_sim,
             z_tx=z_tx, x_rx=x_rx, z_rx=z_rx, mode_plot='envelope', tmin=80, tmax=300, path2plot=fname_save)
m_ij = misfit_function_ij(bscan_data.get_ascan_from_depth(z_tx, x_rx, z_rx), bscan_sim.get_ascan_from_depth(z_tx, x_rx, z_rx),
                          bscan_sim.tspace)
print(m_ij, 1/m_ij)