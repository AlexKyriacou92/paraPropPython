import sys
import numpy as np
from matplotlib import pyplot as pl
import time
import datetime
import h5py
from fitness_function import fitness_correlation
from makeSim import createMatrix
from genetic_functions import initialize_from_analytical

sys.path.append('../')
import paraPropPython as ppp
from receiver import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rx_ranges, create_hdf_bscan, create_tx_signal
from data import create_transmitter_array, bscan, create_rxList

nIndividuals = 10
nprof_0 = np.genfromtxt('start_profiles/parallel-profile-0504_1st-pk.txt')[:,1]
n_prof_initial = initialize_from_analytical(nprof_0, 0.2*np.ones(len(nprof_0)), nIndividuals)

createMatrix(fname_config='config_aletsch.txt', n_prof_initial=n_prof_initial, fname_nmatrix='test-nmatrix.h5', nGenerations = 10)