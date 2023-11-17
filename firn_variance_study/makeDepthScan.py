import os.path
import sys
import numpy as np
import time
import datetime
import h5py
import configparser
import datetime


sys.path.append('../')

import util
import paraPropPython as ppp
from receiver import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rxList_from_file, create_transmitter_array, create_hdf_FT
from data import create_tx_signal, bscan, bscan_rxList, create_hdf_bscan
from data import create_tx_signal_from_file, get_IR_from_config
from data import save_field_to_file, ascan, create_transmitter_array_from_file
sys.path.append('inversion/')
#from objective_functions import misfit_function_ij

def run_field(fname_config, n_profile, z_profile, z_tx, freq, fname_out):
    sim = create_sim(fname_config)
    sim.set_n(nVec=n_profile, zVec=z_profile)  # Set Refractive Index Profile
    sim.set_dipole_source_profile(freq, z_tx)  # Set Source Profile
    sim.set_cw_source_signal(freq)
    sim.do_solver()
    save_field_to_file(sim, fname_out)

def run_ascan_rx(fname_config, n_profile, z_profile, z_tx, freq, fname_hdf, fname_npy):
    sim = create_sim(fname_config)
    rxList = create_rxList_from_file(fname_config)
    nRx = len(rxList)

    txList = create_transmitter_array_from_file(fname_config)

    sim.set_n(nVec=n_profile, zVec=z_profile)  # Set Refractive Index Profile

    ascan_in = ascan()
    ascan_in.load_from_hdf(fname_hdf=fname_hdf)
    tx_signal_in = ascan_in.tx_signal
    tx_spectrum_in = tx_signal_in.get_spectrum()
    freq_space = tx_signal_in.get_freq_space()
    ii_freq = util.findNearest(freq_space, freq)
    print(freq_space, freq)
    '''
    for i in range(len(freq_space)):
        print(i, freq_space[i], freq)
    '''
    amp_ii = tx_spectrum_in[ii_freq]

    ii_tx = util.findNearest(txList, z_tx)
    print('amplitude:', amp_ii, 'index:', ii_freq, 'freq=', freq_space[ii_freq])

    sim.set_dipole_source_profile(centerFreq=freq, depth=z_tx, A=amp_ii)  # Set Source Profile
    sim.set_cw_source_signal(freq=freq)
    sim.do_solver()

    rx_spectrum = np.load(fname_npy, 'r+')
    for ii_rx in range(nRx):
        rx_ii = rxList[ii_rx]
        amp_rx = sim.get_field(x0=rx_ii.x, z0=rx_ii.z)
        print('tx num', ii_tx, 'rx num', ii_rx, 'ii_freq', ii_freq, '\n rx:', rx_ii.x, rx_ii.z, 'amp:', amp_rx)
        rx_spectrum[ii_tx, ii_rx, ii_freq] = amp_rx

'''
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
'''
def depth_scan_impulse(fname_config, n_profile, z_profile, fname_out=None):
    tx_signal = create_tx_signal(fname_config)
    tx_signal.set_impulse()
    f_centre = tx_signal.frequency
    bandwidth = tx_signal.bandwidth
    fmin = f_centre - bandwidth/2
    fmax = f_centre + bandwidth/2
    tx_signal.apply_bandpass(fmin=fmin, fmax=fmax)
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

def depth_scan_impulse_smooth(fname_config, n_profile, z_profile, fname_out=None):
    tx_signal = create_tx_signal(fname_config)
    tx_signal.set_impulse()
    f_centre = tx_signal.frequency
    bandwidth = tx_signal.bandwidth
    fmin = f_centre - bandwidth/2
    fmax = f_centre + bandwidth/2
    tx_signal.apply_bandpass(fmin=fmin, fmax=fmax)
    tx_signal.add_gaussian_noise()
    rxList0 = create_rxList_from_file(fname_config)
    tx_depths = create_transmitter_array(fname_config)

    IR_rx, IR_freq_rx = get_IR_from_config(fname_config, antenna='RX')

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
        sim.do_solver_smooth(rxList, ant_length=1.0, freqMin=tx_signal.freqMin, freqMax=tx_signal.freqMax)
        #sim.do_solver_smooth()
        print('complete')
        tend = time.time()
        for j in range(nReceivers):
            rx_j = rxList[j]
            rx_j.do_impulse_response(IR_freq=IR_freq_rx, IR_data=IR_rx)
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

def depth_scan_impulse_response(fname_config, n_profile, z_profile, fname_out=None):
    tx_signal = create_tx_signal(fname_config)

    tx_signal.set_impulse()
    IR_tx, IR_freq_tx = get_IR_from_config(fname_config, antenna='TX')
    IR_rx, IR_freq_rx = get_IR_from_config(fname_config, antenna='RX')


    tx_signal.do_impulse_response(IR_tx, IR_freq_tx)
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
            rx_j.do_impulse_response(IR_freq=IR_freq_rx, IR_data=IR_rx)
            rx_j.add_gaussian_noise(noise_amplitude=rx_j.noise_amplitude)
            rx_sig = rx_j.get_signal()
            bscan_npy[i,j] = rx_sig
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

def depth_scan_cw(fname_config, n_profile, z_profile, fname_out=None):
    tx_signal = create_tx_signal(fname_config)
    IR_tx, IR_freq_tx = get_IR_from_config(fname_config, antenna='TX')
    IR_rx, IR_freq_rx = get_IR_from_config(fname_config, antenna='RX')

    tx_signal.set_psuedo_fmcw(fmin=tx_signal.freqMin, fmax=tx_signal.freqMax,
                              IR=IR_tx, IR_freq=IR_freq_tx)
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
            rx_j.do_impulse_response(IR_freq=IR_freq_rx, IR_data=IR_rx)
            rx_j.add_gaussian_noise(noise_amplitude=rx_j.noise_amplitude)
            rx_sig = rx_j.get_signal()
            bscan_npy[i,j] = rx_sig
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