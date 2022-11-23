import sys
import numpy as np
import time
import datetime
import h5py

sys.path.append('../')
from transmitter import tx_signal
from data import create_tx_signal
from data import create_rxList, create_hdf_bscan, create_transmitter_array, create_receiver_array
from data import create_sim
from data import bscan
import util
hdf_data = h5py.File('Field-Test-data.h5','r')

fftArray = np.array(hdf_data['fftArray'])
tspace = np.array(hdf_data['tspace'])
txDepths = np.array(hdf_data['txDepths'])
rxDepths = np.array(hdf_data['rxDepths'])
rxRanges = np.array(hdf_data['rxRanges'])

fname_aletsch = '../share/aletsch_glacier_model2.txt'

fname_config = 'config_FFT_compare.txt' # Config file (txt file)
fname_nprofile = fname_aletsch # Refractive index profile (txt file)
fname_output = 'FFT_sim_bscan.h5' # Output file for Bscan (hdf or h5 file)

bscan_simulation = bscan()
bscan_simulation.load_sim('FFT_sim_bscan.h5','r')

bscan_sig = bscan_simulation.bscan_sig