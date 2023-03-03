import sys
import numpy as np
from matplotlib import pyplot as pl
import time
import datetime
import h5py
#from fitness_function import fitness_correlation
sys.path.append('../')
import paraPropPython as ppp
from receiver import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rx_ranges, create_hdf_bscan, create_tx_signal
from data import create_transmitter_array, bscan, create_rxList_from_file

#The Free Parameters and Goodness of Fit are Stored Here

def createMatrix(fname_config, n_prof_initial, z_profile, fname_nmatrix, nGenerations):  # creates matrix
    nProf = len(n_prof_initial)
    nDepths = len(n_prof_initial[0])
    nmatrix_hdf = h5py.File(fname_nmatrix, 'w')

    S_arr = np.zeros((nGenerations, nProf))
    n_matrix = np.zeros((nGenerations, nProf, nDepths))
    n_matrix[0] = n_prof_initial
    nmatrix_hdf.create_dataset('n_profile_matrix', data=n_matrix)
    nmatrix_hdf.create_dataset('z_profile', data=z_profile)

    #nGenes = len(genes_initial)
    #genes_matrix = np.zeros((nGenerations, nProf, ))
    #nmatrix_hdf.create_dataset('genes_matrix', data)

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

    tx_depths = create_transmitter_array(fname_config)
    rxList = create_rxList_from_file(fname_config)
    nmatrix_hdf.create_dataset("source_depths", data=tx_depths)
    nmatrix_hdf.create_dataset('tspace', data=tx_signal.tspace)

    tx_signal.get_gausspulse()
    nmatrix_hdf.create_dataset('signalPulse', data=tx_signal.pulse)
    nmatrix_hdf.create_dataset('signalSpectrum', data=tx_signal.spectrum)

    rxList_positions = np.ones((len(rxList),2))
    for i in range(len(rxList)):
        rx_i = rxList[i]
        rxList_positions[i,0] = rx_i.x
        rxList_positions[i,1] = rx_i.z


    nmatrix_hdf.create_dataset('rxList', data=rxList_positions)
    nmatrix_hdf.close()

def createMatrix2(fname_config, n_prof_initial, genes_initial, z_profile, z_genes, fname_nmatrix,nGenerations):  # creates matrix
    nProf = len(n_prof_initial)
    nDepths = len(n_prof_initial[0])
    nmatrix_hdf = h5py.File(fname_nmatrix, 'w')

    S_arr = np.zeros((nGenerations, nProf))
    n_matrix = np.zeros((nGenerations, nProf, nDepths))
    n_matrix[0] = n_prof_initial
    nmatrix_hdf.create_dataset('n_profile_matrix', data=n_matrix)
    nmatrix_hdf.create_dataset('z_profile', data=z_profile)

    nGenes = len(genes_initial)
    genes_matrix = np.zeros((nGenerations, nProf, nGenes))
    genes_matrix[0] = genes_initial
    nmatrix_hdf.create_dataset('genes_matrix', data=genes_matrix)
    nmatrix_hdf.create_dataset('z_genes', data=z_genes)
    nmatrix_hdf.create_dataset('S_arr', data=S_arr)

    nmatrix_hdf.attrs["nGenerations"] = nGenerations
    nmatrix_hdf.attrs["nIndividuals"] = nProf
    nmatrix_hdf.attrs['nGenes'] = nGenes
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

    tx_depths = create_transmitter_array(fname_config)
    rxList = create_rxList_from_file(fname_config)
    nmatrix_hdf.create_dataset("source_depths", data=tx_depths)
    nmatrix_hdf.create_dataset('tspace', data=tx_signal.tspace)

    tx_signal.get_gausspulse()
    nmatrix_hdf.create_dataset('signalPulse', data=tx_signal.pulse)
    nmatrix_hdf.create_dataset('signalSpectrum', data=tx_signal.spectrum)

    rxList_positions = np.ones((len(rxList), 2))
    for i in range(len(rxList)):
        rx_i = rxList[i]
        rxList_positions[i, 0] = rx_i.x
        rxList_positions[i, 1] = rx_i.z

    nmatrix_hdf.create_dataset('rxList', data=rxList_positions)
    nmatrix_hdf.close()
