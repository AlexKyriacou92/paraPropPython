import os.path
import sys
import numpy as np
import time
import datetime
import h5py
import configparser
from numpy import exp, log
import matplotlib.pyplot as pl
from ku_scripting import *


sys.path.append('../')
from paraPropPython import paraProp as ppp
import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rxList_from_file, create_tx_signal, create_transmitter_array
from data import create_hdf_bscan, bscan_rxList, create_hdf_FT
from data import create_ascan_hdf, ascan
import util

def create_spectrum(fname_config, nprof_data, zprof_data,  fname_output_npy, fname_output_h5):
    tx_signal_out = create_tx_signal(fname_config)
    rxList = create_rxList_from_file(fname_config)
    txList = create_transmitter_array(fname_config)

    nSamples = tx_signal_out.nSamples
    nRX = len(rxList)
    nTX = len(txList)

    #ascan_npy = np.zeros(nTX, nRX, nSamples) #TODO Create Memmp
    spectrum_npy = util.create_memmap(fname_output_npy,
                                   (nTX, nRX, nSamples))
    hdf_ascan = create_ascan_hdf(fname_config=fname_config,
                                 nprof_data=nprof_data,
                                 zprof_data=zprof_data,
                                 fname_output=fname_output_h5)
    return spectrum_npy

def run_scan_impulse(fname_config, fname_output, nprofile_data, zprofile_data, ii_freq, jj_tx):
    tx_signal_out = create_tx_signal(fname_config)
    rxList = create_rxList_from_file(fname_config)
    txList = create_transmitter_array(fname_config)

    nSamples = tx_signal_out.nSamples
    nRX = len(rxList)
    nTX = len(txList)

    tx_signal.set_impulse()
    f_centre = tx_signal.frequency
    bandwidth = tx_signal.bandwidth
    fmin = f_centre - bandwidth/2
    fmax = f_centre + bandwidth/2
    tx_signal.apply_bandpass(fmin=fmin, fmax=fmax)
    tx_signal.add_gaussian_noise()

    spectrum = tx_signal.get_spectrum()
    freq_space = tx_signal.get_freq_space()

    amp_select = spectrum[ii_freq]
    freq_select = freq_space[ii_freq]
    z_tx = txList[jj_tx]

    sim = create_sim(fname_config)
    sim.set_n(nVec=nprofile_data, zVec=zprofile_data)  # Set Refractive Index Profile
    sim.set_dipole_source_profile(centerFreq=freq_select, depth=z_tx, A=amp_select)  # Set Source Profile
    sim.set_cw_source_signal(freq_select)
    sim.do_solver()

    with np.load(fname_output, 'w+') as spectrum_npy:
        for kk_rx in range(nRX):
            rx_kk = rxList[kk_rx]
            x_rx = rx_kk.x
            z_rx = rx_kk.z
            spectrum_npy[jj_tx, kk_rx, ii_freq] = sim.get_field(x0=x_rx,z0=z_rx)

def save_spectrum(fname_output_npy, fname_output_hdf):
    spectrum_npy = np.load(fname_output_npy)
    ascan_out = ascan()
    ascan_out.load_from_hdf(fname_output_hdf)
    ascan_out.load_spectrum(spectrum_npy)

def get_dates(fname_in):
    with  h5py.File(fname_in, 'r') as input_hdf:
        data_matrix = np.array(input_hdf['density'])
    date_arr = data_matrix[1:,0]
    return date_arr