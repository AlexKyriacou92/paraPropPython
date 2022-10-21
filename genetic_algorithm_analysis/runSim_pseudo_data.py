import sys
import numpy as np
from matplotlib import pyplot as pl
import time
import datetime
import h5py
from fitness_function import fitness_correlation
sys.path.append('../')
import paraPropPython as ppp
from receiver import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rx_ranges, create_hdf_bscan, create_tx_signal
from data import create_transmitter_array, bscan, create_rxList
'''
I need this function to run a Bscan from a n_profile_list
'''

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

rx_ranges = create_rx_ranges(fname_config)
tx_depths = create_transmitter_array(fname_config)
nDepths = len(tx_depths)
rx_depths = tx_depths
nRX_x = len(rx_ranges)
nRX_z = len(rx_depths)

bscan_npy = np.zeros((nDepths, nRX_x, nRX_z, tx_signal.nSamples),dtype='complex')

for i in range(nDepths):

    tstart = time.time()
    #=========================================================================================================
    sourceDepth = tx_depths[i]
    sim = create_sim(fname_config)
    sim.set_n(nVec=n_profile, zVec=z_profile) #Set Refractive Index Profile
    sim.set_dipole_source_profile(tx_signal.frequency, sourceDepth)  # Set Source Profile
    sim.set_td_source_signal(tx_signal.pulse, tx_signal.dt) #Set transmitted signal
    rxList = create_rxList(rx_ranges, rx_depths)

    if i == 0:
        output_hdf = create_hdf_bscan(fname=fname_output, sim=sim, tx_signal=tx_signal,
                                      tx_depths=tx_depths, rx_depths=rx_depths, rx_ranges=rx_ranges)

    sim.do_solver(rxList, freqMin=tx_signal.freqMin, freqMax=tx_signal.freqMax)
    ii = 0
    for j in range(nRX_x):
        for k in range(nRX_z):
            rx_jk = rxList[ii]
            bscan_npy[i,j,k,:] = rx_jk.get_signal()
            ii += 1
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