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
if len(sys.argv) != 5:
    print('error! wrong arg number \n enter: python ',sys.argv[0], ' <fname_config> <initial array of pro>')

fname_config = sys.argv[1] #config file
n_prof_initial = sys.argv[2] #the initial list of n_profiles
fname_nmatrix = sys.argv[3] #h5 file -> will contain all n_profiles, and fitness (S) values for each individual from each generation
nGenerations = int(sys.argv[4]) #number of generations
'''

def createMatrix(fname_config, n_prof_initial, z_profile, fname_nmatrix, nGenerations): #creates matrix
    nProf = len(n_prof_initial)
    nDepths = len(n_prof_initial[0])
    nmatrix_hdf = h5py.File(fname_nmatrix,'w')

    S_arr = np.zeros((nGenerations, nProf))
    n_matrix = np.zeros((nGenerations, nProf, nDepths))
    n_matrix[0] = n_prof_initial
    nmatrix_hdf.create_dataset('n_profile_matrix', data=n_matrix)
    nmatrix_hdf.create_dataset('z_profile', data=z_profile)
    nmatrix_hdf.create_dataset('S_arr', data=S_arr)
    nmatrix_hdf.attrs["nGenerations"] = nGenerations
    nmatrix_hdf.attrs["nIndividuals"] = nProf

    sim = create_sim(fname_config)
    nmatrix_hdf.attrs["iceDepth"] = sim.iceDepth
    nmatrix_hdf.attrs["iceLength"] = sim.iceLength
    nmatrix_hdf.attrs["airHeight"] = sim.airHeight
    nmatrix_hdf.attrs["dx"] = sim.dx
    nmatrix_hdf.attrs["dz"] = sim.dz
    
    tx_signal = create_tx_signal(fname_config)
    nmatrix_hdf.attrs["Amplitude"] = tx_signal.amplitude
    nmatrix_hdf.attrs["freqCentral"] = tx_signal.frequency
    nmatrix_hdf.attrs["Bandwidth"] = tx_signal.bandwidth
    nmatrix_hdf.attrs["freqMax"] = tx_signal.freqMax
    nmatrix_hdf.attrs["freqMin"] = tx_signal.freqMin
    nmatrix_hdf.attrs["freqSample"] = tx_signal.fsample
    nmatrix_hdf.attrs["freqNyquist"] = tx_signal.freq_nyq
    nmatrix_hdf.attrs["tCentral"] = tx_signal.t_centre
    nmatrix_hdf.attrs["tSample"] = tx_signal.tmax
    nmatrix_hdf.attrs["dt"] = tx_signal.dt
    nmatrix_hdf.attrs["nSamples"] = tx_signal.nSamples

    rx_ranges = create_rx_ranges(fname_config)
    tx_depths = create_transmitter_array(fname_config)
    rx_depths = tx_depths
    
    nRX_x = len(rx_ranges)
    nRX_z = len(rx_depths)
    rxArray = np.ones((nRX_x, nRX_z, 2))
    for i in range(nRX_x):
        for j in range(nRX_z):
            rxArray[i, j, 0] = rx_ranges[i]
            rxArray[i, j, 1] = rx_depths[j]

    nmatrix_hdf.create_dataset("rxArray", data=rxArray)
    nmatrix_hdf.create_dataset("source_depths", data=tx_depths)
    nmatrix_hdf.create_dataset('tspace', data=tx_signal.tspace)

    tx_signal.get_gausspulse()
    nmatrix_hdf.create_dataset('signalPulse', data=tx_signal.pulse)
    nmatrix_hdf.create_dataset('signalSpectrum', data=tx_signal.spectrum)
    nmatrix_hdf.create_dataset("rx_range", data=rx_ranges)
    nmatrix_hdf.create_dataset("rx_depths", data=rx_depths)
    
    nmatrix_hdf.close()

def create_ref_bscan(fname_config, fname_nprofile):
    tx_signal = create_tx_signal(fname_config)
    tx_signal.get_gausspulse()

    rx_ranges = create_rx_ranges(fname_config)
    tx_depths = create_transmitter_array(fname_config)
    nDepths = len(tx_depths)
    rx_depths = tx_depths
    nRX_x = len(rx_ranges)
    nRX_z = len(rx_depths)

    n_data = np.genfromtxt(fname_nprofile)
    z_profile = n_data[:, 0]
    n_profile = n_data[:, 1]

    print(tx_signal.freqMin, tx_signal.freqMax)

    bscan_npy = np.zeros((nDepths, nRX_x, nRX_z, tx_signal.nSamples), dtype='complex')
    for i in range(nDepths):

        tstart = time.time()
        # =========================================================================================================
        sourceDepth = tx_depths[i]
        print('start solution for ', fname_nprofile, ' z_tx =', sourceDepth, ' m below surface', i + 1, ' out of',
              nDepths)

        sim = create_sim(fname_config)
        sim.set_n(nVec=n_profile, zVec=z_profile)  # Set Refractive Index Profile
        sim.set_dipole_source_profile(tx_signal.frequency, sourceDepth)  # Set Source Profile
        sim.set_td_source_signal(tx_signal.pulse, tx_signal.dt)  # Set transmitted signal
        rxList = create_rxList(rx_ranges, rx_depths)

        sim.do_solver(rxList, freqMin=tx_signal.freqMin, freqMax=tx_signal.freqMax)
        if i == 0:
            output_hdf = create_hdf_bscan(fname=fname_out, sim=sim, tx_signal=tx_signal,
                                          tx_depths=tx_depths, rx_ranges=rx_ranges, rx_depths=rx_depths)

        ii = 0
        for j in range(nRX_x):
            for k in range(nRX_z):
                rx_jk = rxList[ii]
                bscan_npy[i, j, k, :] = rx_jk.get_signal()
                ii += 1
        # ==========================================================================================================
        tend = time.time()
        duration_s = (tend - tstart)
        duration = datetime.timedelta(seconds=duration_s)
        remainder = duration_s * (nDepths - (i + 1))
        completed = round(float(i + 1) / float(nDepths) * 100, 2)
        print(completed, ' % completed')
        print('remaining steps', nDepths - (i + 1), '\nremaining time:', remainder, '\n')
        now = datetime.datetime.now()
        tstamp_now = now.timestamp()
        end_time = datetime.datetime.fromtimestamp(tstamp_now + duration_s)
        print('completion at:', end_time)
        print('')
    output_hdf.create_dataset('bscan_sig', data=bscan_npy)
    output_hdf.close()
