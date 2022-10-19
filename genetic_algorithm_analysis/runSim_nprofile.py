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

if len(sys.argv) != 6:
    print('error! you must enter argument: \npython ' + sys.argv[0] + ' <config.txt> <fname_data.h5> <fname_nprofile_matrix.h5 i_gene j_individual')

fname_config = sys.argv[1]
fname_data = sys.argv[2] # This must contain the date or the psuedo-data -> bscan
fname_n_matrix = sys.argv[3] # I use this to store the results AND the simulation parameters
ii_gene = int(sys.argv[4])
jj_select = int(sys.argv[5])

n_matrix_hdf = h5py.File(fname_n_matrix,'r')
n_profile_matrix = np.array(n_matrix_hdf.get('n_profile_matrix'))
n_profile_ij = n_profile_matrix[ii_gene,jj_select]
z_profile_ij = np.array(n_matrix_hdf.get('z_profile'))

n_matrix_hdf.close()

if jj_select >= len(n_profile_matrix):
    print('error! jj_select must be greater than zero and less than the number of randomized profiles in ', fname_n_matrix)
    print(0, ' < jj_select < ', len(n_profile_matrix))
    sys.exit(-1)
#fname_out = sys.argv[4]

bscan_data = bscan()
bscan_data.load_sim(fname_data)

tx_signal = create_tx_signal(fname_config)
tx_signal.get_gausspulse()

rx_ranges = create_rx_ranges(fname_config)
tx_depths = create_transmitter_array(fname_config)
nDepths = len(tx_depths)
rx_depths = tx_depths
nRX_x = len(rx_ranges)
nRX_z = len(rx_depths)

bscan_npy = np.zeros((nDepths, nRX_x, nRX_z, tx_signal.nSamples),dtype='complex')

print(n_profile_ij)
print('')
print(z_profile_ij)
for i in range(nDepths):

    tstart = time.time()
    #=========================================================================================================
    sourceDepth = tx_depths[i]
    sim = create_sim(fname_config)
    sim.set_n(nVec=n_profile_ij, zVec=z_profile_ij) #Set Refractive Index Profile
    sim.set_dipole_source_profile(tx_signal.frequency, sourceDepth)  # Set Source Profile
    sim.set_td_source_signal(tx_signal.pulse, tx_signal.dt) #Set transmitted signal
    rxList = create_rxList(rx_ranges, rx_depths)

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
    remainder = duration_s * (nDepths - (i+1))
    completed = round(float(i+1)/float(nDepths) * 100,2)
    print(completed, ' % completed')
    print('remaining steps', nDepths - (i+1),'\nremaining time:', remainder,'\n')
    now = datetime.datetime.now()
    tstamp_now = now.timestamp()
    end_time = datetime.datetime.fromtimestamp(tstamp_now + duration_s)
    print('completion at:', end_time)
    print('')

Corr = 0
for i in range(nDepths):
    for j in range(nRX_x):
        for k in range(nRX_z):
            Corr += fitness_correlation(abs(bscan_data.bscan_sig[i,j,k]), abs(bscan_npy[i,j,k]))
S_corr = 1/Corr
print(Corr, S_corr)
n_matrix_hdf = h5py.File(fname_n_matrix,'r+')
S_arr = n_matrix_hdf['S_matrix']
S_arr[ii_gene,jj_select] = S_corr
n_matrix_hdf.close()