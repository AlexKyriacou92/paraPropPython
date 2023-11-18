import sys
import numpy as np
import time
import datetime
import h5py
sys.path.append('../')

import util
import paraPropPython as ppp
from receiver import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rxList_from_file, create_transmitter_array, create_hdf_FT
from data import create_tx_signal, bscan, bscan_rxList, create_hdf_FT

sys.path.append('../genetic_algorithm_analysis/')
from fitness_function import fitness_correlation, fitness_pulse_FT_data

if len(sys.argv) != 4:
    print('error! you must enter argument: \npython ' + sys.argv[0] + ' <config.txt> <nprofile.txt> <foutput.h5>')

fname_config = sys.argv[1]
fname_nprofile = sys.argv[2]
fname_output = sys.argv[3]

data_nprofile = np.genfromtxt(fname_nprofile)
z_profile = data_nprofile[:,0]
n_profile = data_nprofile[:,1]

tx_signal = create_tx_signal(fname_config)
tx_signal.get_gausspulse()

rxList0 = create_rxList_from_file(fname_config)
tx_depths = create_transmitter_array(fname_config)
nDepths = len(tx_depths)
rx_depths = tx_depths
nReceivers = len(rxList0)

bscan_npy = np.zeros((nDepths, nReceivers, tx_signal.nSamples),dtype='complex')

for i in range(nDepths):
    tstart = time.time()
    #=========================================================================================================
    sourceDepth = tx_depths[i]
    sim = create_sim(fname_config)
    sim.set_n(nVec=n_profile, zVec=z_profile) #Set Refractive Index Profile
    sim.set_dipole_source_profile(tx_signal.frequency, sourceDepth)  # Set Source Profile
    sim.set_td_source_signal(tx_signal.pulse, tx_signal.dt) #Set transmitted signal
    rxList = rxList0

    if i == 0:
        output_hdf = create_hdf_FT(fname=fname_output, sim=sim, tx_signal=tx_signal,
                                      tx_depths=tx_depths, rxList=rxList)

    sim.do_solver(rxList, freqMin=tx_signal.freqMin, freqMax=tx_signal.freqMax)
    ii = 0
    for j in range(nReceivers):
        rx_ij = rxList[j]
        bscan_npy[i,j] = rx_ij.get_signal()
    #==========================================================================================================
    tend = time.time()
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


output_hdf.create_dataset('bscan_sig', data=bscan_npy)
output_hdf.close()