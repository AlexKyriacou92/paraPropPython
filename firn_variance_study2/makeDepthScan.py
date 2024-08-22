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

from Askaryan_Signal import create_pulse as create_pulse_askaryan, TeV

def run_field(fname_config, n_profile, z_profile, z_tx, freq, fname_out):
    sim = create_sim(fname_config)
    sim.set_n(nVec=n_profile, zVec=z_profile)  # Set Refractive Index Profile
    sim.set_dipole_source_profile(freq, z_tx)  # Set Source Profile
    sim.set_cw_source_signal(freq)
    sim.do_solver()
    save_field_to_file(sim, fname_out)

def run_ascan_rx(fname_config, n_profile, z_profile, z_tx, freq_in, fname_hdf, fname_npy):
    sim = create_sim(fname_config)
    rxList = create_rxList_from_file(fname_config)
    nRx = len(rxList)

    txList = create_transmitter_array_from_file(fname_config)

    sim.set_n(nVec=n_profile, zVec=z_profile)  # Set Refractive Index Profile

    ascan_in = ascan()
    ascan_in.load_from_hdf(fname_hdf=fname_hdf)
    tx_signal_in = ascan_in.tx_signal

    pulse_tx = tx_signal_in.pulse.real
    spectrum = util.doFFT(np.flip(pulse_tx))

    dt = tx_signal_in.dt
    nSamples = tx_signal_in.nSamples
    freq = tx_signal_in.freq_plus

    ii_freq = util.findNearest(freq, freq_in)
    #print(freq, freq)
    '''
    for i in range(len(freq_space)):
        print(i, freq_space[i], freq)
    '''
    amp_ii = spectrum[ii_freq]

    ii_tx = util.findNearest(txList, z_tx)
    print('amplitude:', amp_ii, 'index:', ii_freq, 'freq=', freq[ii_freq])

    sim.set_dipole_source_profile(centerFreq=freq_in, depth=z_tx, A=amp_ii)  # Set Source Profile
    sim.set_cw_source_signal(freq=freq_in)
    sim.do_solver()

    rx_spectrum = np.load(fname_npy, 'r+')
    for ii_rx in range(nRx):
        rx_ii = rxList[ii_rx]
        amp_rx = sim.get_field(x0=rx_ii.x, z0=rx_ii.z)
        rx_spectrum[ii_tx, ii_rx] = amp_rx
    print('save values', rx_spectrum[:])
    print('Close')

def run_ascan_rx_txt(fname_config, n_profile, z_profile, z_tx, freq_in, fname_hdf, fname_txt):
    sim = create_sim(fname_config)
    rxList = create_rxList_from_file(fname_config)
    nRx = len(rxList)

    txList = create_transmitter_array_from_file(fname_config)

    sim.set_n(nVec=n_profile, zVec=z_profile)  # Set Refractive Index Profile

    ascan_in = ascan()
    ascan_in.load_from_hdf(fname_hdf=fname_hdf)
    tx_signal_in = ascan_in.tx_signal
    pulse_tx = tx_signal_in.pulse.real
    spectrum = util.doFFT(np.flip(pulse_tx))
    freq_plus = tx_signal_in.freq_plus
    ii_freq = util.findNearest(freq_plus, freq_in)
    print(freq_plus, freq_in)

    amp_ii = spectrum[ii_freq]
    print('amplitude:', amp_ii, 'index:', ii_freq, 'freq=', freq_plus[ii_freq])

    sim.set_dipole_source_profile(centerFreq=freq_in, depth=z_tx, A=amp_ii)  # Set Source Profile
    sim.set_cw_source_signal(freq=freq_in)
    sim.do_solver()
    with open(fname_txt, 'w') as fout:
        fout.write('tx_z\t' + str(z_tx) + 'freq_GHz\t' + str(freq_in)+'\n')
        fout.write('rx_x\trx_z\tamp_rx_real\tamp_rx_imag\n')
        for ii_rx in range(nRx):
            rx_ii = rxList[ii_rx]
            amp_rx = sim.get_field(x0=rx_ii.x, z0=rx_ii.z)
            line = str(rx_ii.x) + '\t' + str(rx_ii.z) + '\t' + str(amp_rx.real) + '\t' + str(amp_rx.imag) + '\n'
            fout.write(line)

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

def depth_scan_askaryan(fname_config, n_profile, z_profile, fname_out=None):
    config = configparser.ConfigParser()
    config.read(fname_config)
    ask_config = config['ASKARYAN']

    dtheta_v = float(ask_config['dtheta_v'])
    E_nu_eV = float(ask_config['showerEnergy'])
    E_nu_TeV = E_nu_eV / TeV
    R_alpha = float(ask_config['attenuationEquivalent'])
    tx_signal_out = create_tx_signal(fname_config)
    tspace = tx_signal_out.tspace
    tmax = tx_signal_out.tmax
    t_center = tx_signal_out.t_centre

    pulse_ask_out, tspace_ask = create_pulse_askaryan(Esh=E_nu_TeV, dtheta_v=dtheta_v, R_alpha=R_alpha,
                                                      t_min=-1 * t_center / 1e3, t_max=(tmax - t_center) / 1e3,
                                                      fs=tx_signal_out.fsample * 1e3)
    pulse_ask_interp = np.interp(tspace, tspace_ask, pulse_ask_out)
    tx_signal_out.set_pulse(pulse_ask_interp, tspace)

    tx_signal_out.add_gaussian_noise()
    rxList0 = create_rxList_from_file(fname_config)
    tx_depths = create_transmitter_array(fname_config)

    nDepths = len(tx_depths)
    nReceivers = len(rxList0)

    bscan_npy = np.zeros((nDepths, nReceivers, tx_signal_out.nSamples), dtype='complex')

    print('Running tx scan')
    for i in range(nDepths):
        tstart = time.time()
        sourceDepth = tx_depths[i]
        print('z = ', sourceDepth)

        rxList = rxList0

        sim = create_sim(fname_config)
        sim.set_n(nVec=n_profile, zVec=z_profile)  # Set Refractive Index Profile
        sim.set_dipole_source_profile(tx_signal_out.frequency, sourceDepth)  # Set Source Profile
        sim.set_td_source_signal(tx_signal_out.pulse, tx_signal_out.dt)  # Set transmitted signal
        print('solving PE')
        sim.do_solver(rxList, freqMin=tx_signal_out.freqMin, freqMax=tx_signal_out.freqMax)
        print('complete')
        tend = time.time()
        for j in range(nReceivers):
            rx_j = rxList[j]
            rx_j.add_gaussian_noise(noise_amplitude=rx_j.noise_amplitude)
            rx_sig = rx_j.get_signal()
            bscan_npy[i,j] = rx_sig
        if i == 0 and fname_out != None:
            hdf_output = create_hdf_FT(fname=fname_out, sim=sim,
                                       tx_signal=tx_signal_out, tx_depths=tx_depths, rxList=rxList)

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