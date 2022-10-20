import sys
import numpy as np
from matplotlib import pyplot as pl
import time
import datetime
import h5py
#from fitness_function import fitness_correlation

sys.path.append('../')
'''
import paraPropPython as ppp
from receiver import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rx_ranges, create_hdf_bscan, create_tx_signal
from data import create_transmitter_array, bscan, create_rxList
'''

fname_hdf = sys.argv[1]
ii_gen = int(sys.argv[2])
jj_ind = int(sys.argv[3])

n_matrix_hdf = h5py.File(fname_hdf, 'r')
S_arr = n_matrix_hdf['S_arr']
#n_profile_matrix = np.array(n_matrix_hdf.get('n_profile_matrix'))

print(S_arr[ii_gen][jj_ind])