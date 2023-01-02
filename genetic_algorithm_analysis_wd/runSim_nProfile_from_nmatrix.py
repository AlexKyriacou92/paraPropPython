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
print('Running, ', sys.argv[0])

if len(sys.argv) == 6:
    fname_config = sys.argv[1]  # The Config File -> sys.argv[1]
    fname_n_matrix = sys.argv[2]  # I use this to store the results AND the simulation parameters sys.argv[3]
    ii_generation = int(sys.argv[3])  # The Generation Number of the n_profile sys.argv[4]
    jj_select = int(sys.argv[4])  # The individual number from that Generation sys.argv[5]
    fname_out = sys.argv[5]
else:
    print('incorrect arg number: ', len(sys.argv), sys.argv)
    sys.exit()
#==============================================

n_matrix_hdf = h5py.File(fname_n_matrix,'r') #The matrix holding n_profiles
n_profile_matrix = np.array(n_matrix_hdf.get('n_profile_matrix'))
n_profile_ij = n_profile_matrix[ii_generation,jj_select] #The Individual (n profile) contains genes (n values per z)
z_profile_ij = np.array(n_matrix_hdf.get('z_profile'))
n_matrix_hdf.close()

nGenes = len(n_profile_ij)
#==============================================

tx_signal = create_tx_signal(fname_config)
tx_signal.get_gausspulse()

rxList0 = create_rxList_from_file(fname_config)
tx_depths = create_transmitter_array(fname_config)
nDepths = len(tx_depths)
nReceivers = len(rxList0)

bscan_npy = np.zeros((nDepths, nReceivers, tx_signal.nSamples),dtype='complex')

print('Running tx scan')
for i in range(nDepths):
    tstart = time.time()
    sourceDepth = tx_depths[i]
    print('z = ', sourceDepth)

    rxList = rxList0

    sim = create_sim(fname_config)
    sim.set_n(nVec=n_profile_ij, zVec=z_profile_ij) #Set Refractive Index Profile
    sim.set_dipole_source_profile(tx_signal.frequency, sourceDepth)  # Set Source Profile
    sim.set_td_source_signal(tx_signal.pulse, tx_signal.dt) #Set transmitted signal
    print('solving PE')
    sim.do_solver(rxList, freqMin=tx_signal.freqMin, freqMax=tx_signal.freqMax)
    print('complete')
    tend = time.time()
    for j in range(nReceivers):
        rx_j = rxList[j]
        bscan_npy[i,j] = rx_j.get_signal()

    if i == 0:
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
hdf_output.create_dataset('bscan_sig', data=bscan_npy)