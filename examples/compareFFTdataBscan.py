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

#Create transmitted signal:
tx_signal = create_tx_signal(fname_config)
tx_signal.get_gausspulse()

#Create array fo transmitter and receivers:
tx_depths = create_transmitter_array(fname_config)
rx_ranges, rx_depths = create_receiver_array(fname_config)

nDepths = len(tx_depths)
nRX_x = len(rx_ranges)
nRX_z = len(rx_depths)

#Create Memmap to save data
bscan_npy = np.zeros((nDepths, nRX_x, nRX_z, tx_signal.nSamples),dtype='complex')

#Open N-Profile
profile_data = np.genfromtxt(fname_nprofile)
z_profile = profile_data[:,0]
n_profile = profile_data[:,1]


for i in range(nDepths):
    tstart = time.time()

    #=========================================================================================================
    sourceDepth = tx_depths[i]
    print(i, sourceDepth)
    sim = create_sim(fname_config)
    sim.set_n(nVec=n_profile, zVec=z_profile) #Set Refractive Index Profile
    sim.set_dipole_source_profile(tx_signal.frequency, sourceDepth)  # Set Source Profile
    sim.set_td_source_signal(tx_signal.pulse, tx_signal.dt) #Set transmitted signal
    rxList = create_rxList(rx_ranges, rx_depths)

    if i == 0:
        output_hdf = create_hdf_bscan(fname=fname_output, sim=sim, tx_signal=tx_signal,
                                      tx_depths=tx_depths, rx_depths=rx_depths, rx_ranges=rx_ranges)

    sim.do_solver(rxList)
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