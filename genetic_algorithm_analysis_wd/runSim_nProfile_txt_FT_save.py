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
from data import create_tx_signal, bscan

sys.path.append('../genetic_algorithm_analysis/')
from fitness_function import fitness_correlation, fitness_pulse_FT_data


if len(sys.argv) != 5:
    print('Wrong argument number:', len(sys.argv), ' should be: 6 or 7')
    print('you must enter argument: \npython ' + sys.argv[0] + ' <config.txt> <fname_data.h5> <fname_nprofile.txt <fname_out?>')
    sys.exit()

fname_config = sys.argv[1]  # The Config File -> sys.argv[1]
fname_data = sys.argv[2]  # This must contain the date or the psuedo-data -> bscan, sys.argv[2]
fname_nprofile = sys.argv[3]  # I use this to store the results AND the simulation parameters sys.argv[3]
fname_out = sys.argv[4]

#==============================================
nprofile_data = np.genfromtxt(fname_nprofile)
n_profile_ij = nprofile_data[:,1] #The Individual (n profile) contains genes (n values per z)
z_profile_ij = nprofile_data[:,0]

nGenes = len(n_profile_ij)
#==============================================

#Field Test Data
hdf_data = h5py.File(fname_data, 'r')
rxRanges_data = np.array(hdf_data['rxRanges'])
rxDepths_data = np.array(hdf_data['rxDepths'])
fftArray_data = np.array(hdf_data['fftArray'])
txDepths_data = np.array(hdf_data['txDepths'])
tspace_data = np.array(hdf_data['tspace'])
nData = len(fftArray_data)
#==============================================

tx_signal = create_tx_signal(fname_config)
tx_signal.get_gausspulse()

rxList0 = create_rxList_from_file(fname_config)
tx_depths = create_transmitter_array(fname_config)
nDepths = len(tx_depths)
nReceivers = len(rxList0)

bscan_npy = np.zeros((nDepths, nReceivers, tx_signal.nSamples),dtype='complex')

print(n_profile_ij)
for i in range(nDepths):
    tstart = time.time()

    sourceDepth = tx_depths[i]
    rxList = rxList0

    sim = create_sim(fname_config)
    sim.set_n(nVec=n_profile_ij, zVec=z_profile_ij) #Set Refractive Index Profile
    sim.set_dipole_source_profile(tx_signal.frequency, sourceDepth)  # Set Source Profile
    sim.set_td_source_signal(tx_signal.pulse, tx_signal.dt) #Set transmitted signal

    if i == 0:
        if fname_out != None:
            hdf_output = create_hdf_FT(fname=fname_out, sim=sim,
                                       tx_signal=tx_signal, tx_depths=tx_depths, rxList=rxList)

    rx_0 = rxList[0]
    sim.do_solver(rxList, freqMin=tx_signal.freqMin, freqMax=tx_signal.freqMax)
    tend = time.time()
    for j in range(nReceivers):
        rx_j = rxList[j]
        bscan_npy[i,j] = rx_j.get_signal()

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

for i in range(nDepths):
    z_tx_sim = tx_depths[i]
    for j in range(nReceivers):
        rx_j = rxList0[j]
        for k in range(nData):
            x_rx_data = rxRanges_data[k]
            z_rx_data = rxDepths_data[k]
            z_tx_data = txDepths_data[k]

            sigFFT = fftArray_data[k]

            dX_rx = abs(x_rx_data - rx_j.x)
            dZ_rx = abs(z_rx_data - rx_j.z)
            dZ_tx = abs(z_tx_data - z_tx_sim)

            if dX_rx < 1.0 and dZ_rx < 0.1 and dZ_tx < 0.1:
                sig_sim = bscan_npy[i,j]
                S_corr_ijk = fitness_pulse_FT_data(sig_sim = sig_sim, sig_data=sigFFT)
                S_corr += S_corr_ijk

print(S_corr)
hdf_output.attrs['S_corr'] = S_corr
hdf_output.close()