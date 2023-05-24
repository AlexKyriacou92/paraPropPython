import os.path
import sys
import numpy as np
import time
import datetime
import h5py
import configparser
import datetime

from objective_functions import fitness_function, misfit_function_ij

sys.path.append('../')

import util
import paraPropPython as ppp
from receiver import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rxList_from_file, create_transmitter_array, create_hdf_FT
from data import create_tx_signal, bscan, bscan_rxList, create_hdf_bscan
from data import create_tx_signal_from_file

def ascan(fname_config, n_profile, z_profile, z_tx, x_rx, z_rx): #TODO: Add Output File
    tx_signal = create_tx_signal(fname_config)
    tx_signal.get_gausspulse()
    tx_signal.add_gaussian_noise()
    rxList = []
    rx_1 = rx(x=x_rx, z=z_rx)
    sourceDepth = z_tx
    rxList.append(rx_1)

    sim = create_sim(fname_config)
    sim.set_n(nVec=n_profile, zVec=z_profile)  # Set Refractive Index Profile
    sim.set_dipole_source_profile(tx_signal.frequency, sourceDepth)  # Set Source Profile
    sim.set_td_source_signal(tx_signal.pulse, tx_signal.dt)  # Set transmitted signal
    print('solving PE')
    sim.do_solver(rxList, freqMin=tx_signal.freqMin, freqMax=tx_signal.freqMax)

    rx_out = rxList[0]
    signal_out = rx_out.get_signal()
    return signal_out

def depth_scan(fname_config, n_profile, z_profile, fname_out=None):
    tx_signal = create_tx_signal(fname_config)
    tx_signal.get_gausspulse()
    tx_signal.add_gaussian_noise()
    rxList0 = create_rxList_from_file(fname_config)
    tx_depths = create_transmitter_array(fname_config)

    nDepths = len(tx_depths)
    nReceivers = len(rxList0)

    bscan_npy = np.zeros((nDepths, nReceivers, tx_signal.nSamples), dtype='complex')

    print('Running tx scan')
    for i in range(nDepths):
        tstart = time.time()
        sourceDepth = tx_depths[i]
        print('z = ', sourceDepth)

        rxList = rxList0

        sim = create_sim(fname_config)
        sim.set_n(nVec=n_profile, zVec=z_profile)  # Set Refractive Index Profile
        sim.set_dipole_source_profile(tx_signal.frequency, sourceDepth)  # Set Source Profile
        sim.set_td_source_signal(tx_signal.pulse, tx_signal.dt)  # Set transmitted signal
        print('solving PE')
        sim.do_solver(rxList, freqMin=tx_signal.freqMin, freqMax=tx_signal.freqMax)
        print('complete')
        tend = time.time()
        for j in range(nReceivers):
            rx_j = rxList[j]
            rx_j.add_gaussian_noise(noise_amplitude=rx_j.noise_amplitude)
            bscan_npy[i, j] = rx_j.get_signal()

        if i == 0 and fname_out != None:
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
    if fname_out != None:
        hdf_output.create_dataset('bscan_sig', data=bscan_npy)
        hdf_output.close()

    return bscan_npy

def depth_scan_from_txt(fname_config, fname_nprofile, fname_out=None):
    n_profile, z_profile = util.get_profile_from_file(fname_nprofile)
    bscan_npy = depth_scan(fname_config=fname_config, n_profile=n_profile, z_profile=z_profile, fname_out=fname_out)
    return bscan_npy

def depth_scan_from_hdf(fname_config, fname_n_matrix, ii_generation, jj_select, fname_pseudo = None, fname_out=None):
    n_matrix_hdf = h5py.File(fname_n_matrix,'r') #The matrix holding n_profiles
    fname_n_matrix_npy = fname_n_matrix[:-3] + '.npy'
    fname_misfit_npy = fname_n_matrix[:-3] + '_misfit.npy'
    n_profile_matrix = np.array(n_matrix_hdf.get('n_profile_matrix'))
    n_profile_ij = n_profile_matrix[ii_generation,jj_select] #The Individual (n profile) contains genes (n values per z)
    z_profile_ij = np.array(n_matrix_hdf.get('z_profile'))
    n_matrix_hdf.close()

    config = configparser.ConfigParser()
    config.read(fname_config)

    # Set up Simulation Output Files
    print('set up output files')
    fitness_mode = config['GA']['Fitness']

    bscan_npy = depth_scan(fname_config=fname_config, n_profile=n_profile_ij, z_profile=z_profile_ij, fname_out=fname_out)

    if fname_pseudo != None:
        bscan_pseudo = bscan_rxList()
        bscan_pseudo.load_sim(fname_pseudo)
        tspace = bscan_pseudo.tspace
        nDepths = len(bscan_npy)
        nReceivers = len(bscan_npy[0])
        if os.path.isfile(fname_misfit_npy) == True:
            misfit_arr = np.load(fname_misfit_npy,'r+')
        misfit_total = 0
        for i in range(nDepths):
            for j in range(nReceivers):
                m_ij = misfit_function_ij(sig_data=bscan_pseudo.get_ascan(i,j),
                                       sig_sim=bscan_npy[i,j],
                                       tspace=tspace, mode=fitness_mode, tmin=100, tmax=300) #TODO: Add ability to change tmin and tmax
                if os.path.isfile(fname_misfit_npy) == True:
                    misfit_arr[ii_generation, jj_select, i, j] = m_ij
                misfit_total += m_ij
        misfit_total /= (2. * float(nDepths)*float(nReceivers))
        #print('Misfit total:', misfit_total, misfit_arr[ii_generation, jj_select])
        S_corr = 1/misfit_total
        print('S=', S_corr)
        S_arr_out = np.load(fname_n_matrix_npy, 'r+')
        S_arr_out[ii_generation, jj_select] = S_corr #WRITE TO NPY FILE


def depth_scan_from_hdf_data(fname_config, fname_n_matrix, ii_generation, jj_select, fname_data, fname_out=None):
    n_matrix_hdf = h5py.File(fname_n_matrix,'r') #The matrix holding n_profiles
    fname_n_matrix_npy = fname_n_matrix[:-3] + '.npy'
    fname_misfit_npy = fname_n_matrix[:-3] + '_misfit.npy'
    n_profile_matrix = np.array(n_matrix_hdf.get('n_profile_matrix'))
    n_profile_ij = n_profile_matrix[ii_generation,jj_select] #The Individual (n profile) contains genes (n values per z)
    z_profile_ij = np.array(n_matrix_hdf.get('z_profile'))
    n_matrix_hdf.close()

    bscan_npy = depth_scan(fname_config=fname_config, n_profile=n_profile_ij, z_profile=z_profile_ij, fname_out=fname_out)
    bscan_sim = bscan_rxList()
    bscan_sim.setup_from_config(fname_config=fname_config, n_profile=n_profile_ij, z_profile=z_profile_ij, bscan_npy=bscan_npy)
    nSamples = bscan_sim.nSamples
    rxList = bscan_sim.rxList
    freq_min = bscan_sim.tx_signal.freqMin
    freq_max = bscan_sim.tx_signal.freqMax

    hdf_data = h5py.File(fname_data, 'r')
    fftArray = np.array(hdf_data['fftArray'])
    freqList = np.array(hdf_data['freqList']) / 1e9 #Note FT Data is defined in s/Hz, while paraProp uses ns/GHz
    rxDepths = np.array(hdf_data['rxDepths'])
    rxRanges = np.array(hdf_data['rxRanges'])
    tspace_data = np.array(hdf_data['tspace']) * 1e9 #Note FT Data is defined in s/Hz, while paraProp uses ns/GHz
    txDepths = np.array(hdf_data['txDepths'])
    hdf_data.close()
    nMeasurements = len(fftArray)

    R_data = np.unique(rxRanges)
    print(R_data)
    config = configparser.ConfigParser()
    config.read(fname_config)
    fitness_mode = config['GA']['Fitness']

    #TODO: Add Ability to Select Frequency
    bscan_arr_sim = np.zeros((nMeasurements, nSamples))
    tx_signal = bscan_sim.tx_signal
    tx_pulse = abs(tx_signal.pulse)
    ii_max = np.argmax(tx_pulse)
    for i in range(nMeasurements):
        #freq_i = freqList[i]
        #if freq_i >= freq_min and freq_i <= freq_max:
        xRx_data = rxRanges[i]
        zTx_data = txDepths[i]
        zRx_data = rxDepths[i]

        ascan_sim = bscan_sim.get_ascan_from_depth(z_tx=zTx_data, x_rx=xRx_data, z_rx=zRx_data)
        ascan_sim = np.roll(ascan_sim, -ii_max)
        bscan_arr_sim[i] = ascan_sim
    Chi_total = 0
    for i in range(nMeasurements):
        #TODO: Ensure Equal Array Sizes
        #TODO: Calculate Misfit between Data and Simul
        ascan_data = fftArray[i]
        ascan_sim = bscan_arr_sim[i]
        chi_ij = misfit_function_ij(ascan_data, ascan_sim, tspace_data, mode=fitness_mode) #TODO Test
        Chi_total += chi_ij
    Chi_total /= float(nMeasurements)
    S_corr = 1/Chi_total

    print('S=', S_corr)
    S_arr_out = np.load(fname_n_matrix_npy, 'r+')
    S_arr_out[ii_generation, jj_select] = S_corr #WRITE TO NPY FILE

def ascan_from_hdf(fname_config, fname_n_matrix, ii_generation, jj_select, z_tx, x_rx, z_rx):
    n_matrix_hdf = h5py.File(fname_n_matrix,'r') #The matrix holding n_profiles
    fname_n_matrix_npy = fname_n_matrix[:-3] + '.npy'
    fname_misfit_npy = fname_n_matrix[:-3] + '_misfit.npy'
    n_profile_matrix = np.array(n_matrix_hdf.get('n_profile_matrix'))
    n_profile_ij = n_profile_matrix[ii_generation,jj_select] #The Individual (n profile) contains genes (n values per z)
    z_profile_ij = np.array(n_matrix_hdf.get('z_profile'))
    n_matrix_hdf.close()

    return ascan(fname_config=fname_config, n_profile = n_profile_ij, z_profile = z_profile_ij,
                 z_tx=z_tx, x_rx=x_rx, z_rx=z_rx)

#======================================================================================================

def depth_scan_IR(fname_config, n_profile, z_profile, fname_out=None):
    tx_signal = create_tx_signal_from_file(fname_config)
    #tx_signal.get_gausspulse()
    #tx_signal.add_gaussian_noise()
    rxList0 = create_rxList_from_file(fname_config)
    tx_depths = create_transmitter_array(fname_config)

    nDepths = len(tx_depths)
    nReceivers = len(rxList0)

    bscan_npy = np.zeros((nDepths, nReceivers, tx_signal.nSamples), dtype='complex')

    print('Running tx scan')
    for i in range(nDepths):
        tstart = time.time()
        sourceDepth = tx_depths[i]
        print('z = ', sourceDepth)

        rxList = rxList0

        sim = create_sim(fname_config)
        sim.set_n(nVec=n_profile, zVec=z_profile)  # Set Refractive Index Profile
        sim.set_dipole_source_profile(tx_signal.frequency, sourceDepth)  # Set Source Profile
        sim.set_td_source_signal(tx_signal.pulse, tx_signal.dt)  # Set transmitted signal
        print('solving PE')
        sim.do_solver(rxList, freqMin=tx_signal.freqMin, freqMax=tx_signal.freqMax)
        print('complete')
        tend = time.time()
        for j in range(nReceivers):
            rx_j = rxList[j]
            rx_j.add_gaussian_noise(noise_amplitude=rx_j.noise_amplitude)
            bscan_npy[i, j] = rx_j.get_signal()

        if i == 0 and fname_out != None:
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
    if fname_out != None:
        hdf_output.create_dataset('bscan_sig', data=bscan_npy)
        hdf_output.close()

    return bscan_npy

def depth_scan_from_txt_IR(fname_config, fname_nprofile, fname_out=None):
    n_profile, z_profile = util.get_profile_from_file(fname_nprofile)
    bscan_npy = depth_scan_IR(fname_config=fname_config, n_profile=n_profile, z_profile=z_profile, fname_out=fname_out)
    return bscan_npy

def depth_scan_from_hdf_IR(fname_config, fname_n_matrix, ii_generation, jj_select, fname_pseudo = None, fname_out=None):
    n_matrix_hdf = h5py.File(fname_n_matrix,'r') #The matrix holding n_profiles
    fname_n_matrix_npy = fname_n_matrix[:-3] + '.npy'
    fname_misfit_npy = fname_n_matrix[:-3] + '_misfit.npy'
    n_profile_matrix = np.array(n_matrix_hdf.get('n_profile_matrix'))
    n_profile_ij = n_profile_matrix[ii_generation,jj_select] #The Individual (n profile) contains genes (n values per z)
    z_profile_ij = np.array(n_matrix_hdf.get('z_profile'))
    n_matrix_hdf.close()

    config = configparser.ConfigParser()
    config.read(fname_config)

    # Set up Simulation Output Files
    print('set up output files')
    fitness_mode = config['GA']['Fitness']

    bscan_npy = depth_scan_IR(fname_config=fname_config, n_profile=n_profile_ij, z_profile=z_profile_ij, fname_out=fname_out)

    if fname_pseudo != None:
        bscan_pseudo = bscan_rxList()
        bscan_pseudo.load_sim(fname_pseudo)
        tspace = bscan_pseudo.tspace
        nDepths = len(bscan_npy)
        nReceivers = len(bscan_npy[0])
        if os.path.isfile(fname_misfit_npy) == True:
            misfit_arr = np.load(fname_misfit_npy,'r+')
        misfit_total = 0
        for i in range(nDepths):
            for j in range(nReceivers):
                m_ij = misfit_function_ij(sig_data=bscan_pseudo.get_ascan(i,j),
                                       sig_sim=bscan_npy[i,j],
                                       tspace=tspace, mode=fitness_mode, tmin=100, tmax=300) #TODO: Add ability to change tmin and tmax
                if os.path.isfile(fname_misfit_npy) == True:
                    misfit_arr[ii_generation, jj_select, i, j] = m_ij
                misfit_total += m_ij
        misfit_total /= (2. * float(nDepths)*float(nReceivers))
        #print('Misfit total:', misfit_total, misfit_arr[ii_generation, jj_select])
        S_corr = 1/misfit_total
        print('S=', S_corr)
        S_arr_out = np.load(fname_n_matrix_npy, 'r+')
        S_arr_out[ii_generation, jj_select] = S_corr #WRITE TO NPY FILE


def depth_scan_from_hdf_data_IR(fname_config, fname_n_matrix, ii_generation, jj_select, fname_data, fname_out=None):
    n_matrix_hdf = h5py.File(fname_n_matrix,'r') #The matrix holding n_profiles
    fname_n_matrix_npy = fname_n_matrix[:-3] + '.npy'
    fname_misfit_npy = fname_n_matrix[:-3] + '_misfit.npy'
    n_profile_matrix = np.array(n_matrix_hdf.get('n_profile_matrix'))
    n_profile_ij = n_profile_matrix[ii_generation,jj_select] #The Individual (n profile) contains genes (n values per z)
    z_profile_ij = np.array(n_matrix_hdf.get('z_profile'))
    n_matrix_hdf.close()

    bscan_npy = depth_scan_IR(fname_config=fname_config, n_profile=n_profile_ij, z_profile=z_profile_ij, fname_out=fname_out)
    bscan_sim = bscan_rxList()
    bscan_sim.setup_from_config(fname_config=fname_config, n_profile=n_profile_ij, z_profile=z_profile_ij, bscan_npy=bscan_npy)
    tx_signal = create_tx_signal_from_file(fname_config)

    nSamples = bscan_sim.nSamples
    rxList = bscan_sim.rxList
    freq_min = bscan_sim.tx_signal.freqMin
    freq_max = bscan_sim.tx_signal.freqMax

    hdf_data = h5py.File(fname_data, 'r')
    fftArray = np.array(hdf_data['fftArray'])
    freqList = np.array(hdf_data['freqList']) / 1e9 #Note FT Data is defined in s/Hz, while paraProp uses ns/GHz
    rxDepths = np.array(hdf_data['rxDepths'])
    rxRanges = np.array(hdf_data['rxRanges'])
    tspace_data = np.array(hdf_data['tspace']) * 1e9 #Note FT Data is defined in s/Hz, while paraProp uses ns/GHz
    txDepths = np.array(hdf_data['txDepths'])
    hdf_data.close()
    nMeasurements = len(fftArray)

    R_data = np.unique(rxRanges)
    print(R_data)
    config = configparser.ConfigParser()
    config.read(fname_config)
    fitness_mode = config['GA']['Fitness']

    #TODO: Add Ability to Select Frequency
    bscan_arr_sim = np.zeros((nMeasurements, nSamples))
    for i in range(nMeasurements):
        #freq_i = freqList[i]
        #if freq_i >= freq_min and freq_i <= freq_max:
        xRx_data = rxRanges[i]
        zTx_data = txDepths[i]
        zRx_data = rxDepths[i]

        ascan_sim = bscan_sim.get_ascan_from_depth(z_tx=zTx_data, x_rx=xRx_data, z_rx=zRx_data)
        bscan_arr_sim[i] = ascan_sim

    Chi_total = 0

    tx_pulse = tx_signal.pulse
    ii_max = np.argmax(abs(tx_pulse))
    for i in range(nMeasurements):
        #TODO: Ensure Equal Array Sizes
        #TODO: Calculate Misfit between Data and Simul
        ascan_data = fftArray[i]
        ascan_sim = bscan_arr_sim[i]
        ascan_sim = np.roll(ascan_sim, -ii_max) #adjust for delay of pulse
        chi_ij = misfit_function_ij(ascan_data, ascan_sim, tspace_data, mode=fitness_mode) #TODO Test
        Chi_total += chi_ij
    Chi_total /= float(nMeasurements)
    S_corr = 1/Chi_total

    print('S=', S_corr)
    S_arr_out = np.load(fname_n_matrix_npy, 'r+')
    S_arr_out[ii_generation, jj_select] = S_corr #WRITE TO NPY FILE

def ascan_from_hdf_IR(fname_config, fname_n_matrix, ii_generation, jj_select, z_tx, x_rx, z_rx):
    n_matrix_hdf = h5py.File(fname_n_matrix,'r') #The matrix holding n_profiles
    fname_n_matrix_npy = fname_n_matrix[:-3] + '.npy'
    fname_misfit_npy = fname_n_matrix[:-3] + '_misfit.npy'
    n_profile_matrix = np.array(n_matrix_hdf.get('n_profile_matrix'))
    n_profile_ij = n_profile_matrix[ii_generation,jj_select] #The Individual (n profile) contains genes (n values per z)
    z_profile_ij = np.array(n_matrix_hdf.get('z_profile'))
    n_matrix_hdf.close()

    return ascan(fname_config=fname_config, n_profile = n_profile_ij, z_profile = z_profile_ij,
                 z_tx=z_tx, x_rx=x_rx, z_rx=z_rx)