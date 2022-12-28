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

from fitness_function import fitness_correlation, fitness_pulse_FT_data

'''
if len(sys.argv) != 6:
    print('error! you must enter argument: \npython ' + sys.argv[0] + ' <config.txt> <fname_data.h5> <fname_nprofile_matrix.h5 i_gene j_individual')
'''
if len(sys.argv) == 6:
    fname_config = sys.argv[1] #The Config File -> sys.argv[1]
    fname_pseudo_data = sys.argv[2] # This must contain the date or the psuedo-data -> bscan, sys.argv[2]
    fname_n_matrix = sys.argv[3] # I use this to store the results AND the simulation parameters sys.argv[3]
    ii_generation = int(sys.argv[4]) #The Generation Number of the n_profile sys.argv[4]
    jj_select = int(sys.argv[5]) #The individual number from that Generation sys.argv[5]
    fname_out = None
elif len(sys.argv) == 7:
    fname_config = sys.argv[1]  # The Config File -> sys.argv[1]
    fname_pseudo_data = sys.argv[2]  # This must contain the date or the psuedo-data -> bscan, sys.argv[2]
    fname_n_matrix = sys.argv[3]  # I use this to store the results AND the simulation parameters sys.argv[3]
    ii_generation = int(sys.argv[4])  # The Generation Number of the n_profile sys.argv[4]
    jj_select = int(sys.argv[5])  # The individual number from that Generation sys.argv[5]
    fname_out = sys.argv[6]
#==============================================

print(fname_pseudo_data, os.path.isfile(fname_pseudo_data))
n_matrix_hdf = h5py.File(fname_n_matrix,'r') #The matrix holding n_profiles
n_profile_matrix = np.array(n_matrix_hdf.get('n_profile_matrix'))
n_profile_ij = n_profile_matrix[ii_generation,jj_select] #The Individual (n profile) contains genes (n values per z)
z_profile_ij = np.array(n_matrix_hdf.get('z_profile'))
n_matrix_hdf.close()

nGenes = len(n_profile_ij)

#==============================================
# Pseudo Data
bscan_pseudo_data = bscan_rxList()
bscan_pseudo_data.load_sim(fname_pseudo_data)
#==============================================

#==============================================

tx_signal = create_tx_signal(fname_config)
tx_signal.get_gausspulse()

rxList0 = create_rxList_from_file(fname_config)
tx_depths = create_transmitter_array(fname_config)
nDepths = len(tx_depths)
nReceivers = len(rxList0)

bscan_npy = np.zeros((nDepths, nReceivers, tx_signal.nSamples),dtype='complex')

for i in range(nDepths):
    tstart = time.time()

    sourceDepth = tx_depths[i]
    rxList = rxList0

    sim = create_sim(fname_config)
    sim.set_n(nVec=n_profile_ij, zVec=z_profile_ij) #Set Refractive Index Profile
    sim.set_dipole_source_profile(tx_signal.frequency, sourceDepth)  # Set Source Profile
    sim.set_td_source_signal(tx_signal.pulse, tx_signal.dt) #Set transmitted signal

    sim.do_solver(rxList, freqMin=tx_signal.freqMin, freqMax=tx_signal.freqMax)
    tend = time.time()
    for j in range(nReceivers):
        rx_j = rxList[j]
        bscan_npy[i,j] = rx_j.get_signal()

    if i == 0:
        if fname_out != None:
            hdf_output = create_hdf_FT(fname=fname_out, sim=sim,
                                       tx_signal=tx_signal, tx_depths=tx_depths, rxList=rxList)

    duration_s = (tend - tstart)
    duration = datetime.timedelta(seconds=duration_s)
    remainder_s = duration_s * (nDepths - (i + 1))
    remainder = datetime.timedelta(seconds=remainder_s)

    completed = round(float(i + 1) / float(nDepths) * 100, 2)
    print(completed, ' % completed, duration:', duration)
    print('remaining steps', nDepths - (i + 1), '\nremaining time:', remainder, '\n')
    now = datetime.datetime.now()
    tstamp_now = now.timestamp()
    end_time = datetime.datetime.fromtimestamp(tstamp_now + remainder_s)
    print('completion at:', end_time)
    print('')

S_corr = 0
if fname_out != None:
    hdf_output.create_dataset('bscan_sig', data=bscan_npy)

#Estimate Weights
pseudodata_inv_weights = np.ones((nDepths, nReceivers))
sim_inv_weights = np.ones((nDepths, nReceivers))

config = configparser.ConfigParser()
weighting_str = config['GA']['Weighting']
fitness_mode = config['GA']['Fitness']
if fitness_mode != 'Correlation' or fitness_mode != 'Difference':
    print('Warning, fitness mode is not set to Correlation or Difference -> code with default to Correlation or Difference')

weighting_bool = False
if weighting_str == 'True':
    weighting_bool = True
elif weighting_str == 'False':
    weighting_bool = False

print('Weighting set to ', weighting_bool)

if weighting_bool == True:
    pseudodata_weights = np.zeros((nDepths, nReceivers))
    sim_weights = np.zeros((nDepths, nReceivers))
    W_pseudodata = 0
    W_sim = 0
    for i in range(nDepths):
        for j in range(nReceivers):
            sim_weights[i,j] = sum(abs(bscan_npy[i,j]))
            W_sim += sim_weights[i,j]
            sig_pseudodata = sum(abs(bscan_pseudo_data[i,j]))
            W_pseudodata += pseudodata_weights[i,j]

    data_inv_weights = W_pseudodata/pseudodata_weights
    sim_inv_weights = W_sim/sim_weights

for i in range(nDepths):
    z_tx_sim = tx_depths[i]
    for j in range(nReceivers):
        rx_j = rxList0[j]
        sig_pseudodata = bscan_pseudo_data.get_ascan(i,j)
        sig_sim = bscan_npy[i,j]
        S_corr_ijk = fitness_pulse_FT_data(sig_sim=sig_sim, sig_data=sig_pseudodata, mode=fitness_mode)
        S_corr += S_corr_ijk

print(S_corr)
n_matrix_hdf = h5py.File(fname_n_matrix,'r+')
S_arr = n_matrix_hdf['S_arr']
S_arr[ii_generation,jj_select] = S_corr
n_matrix_hdf.close()