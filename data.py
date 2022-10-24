import h5py
import paraPropPython as ppp
from transmitter import tx_signal
from receiver import receiver as rx
import numpy as np
from numpy.lib.format import open_memmap

import util
import os
from matplotlib import pyplot as pl
import configparser
import argparse

def create_sim(fname_config): #Creates Simulation from config file using parser
    '''
    Creates paraProp simulation object (see paraPropPython.py) object from config file

    Note -> paraProp object defines geometry only -> source, signal, receivers and n-profile has to be defined afterwards
    '''
    config = configparser.ConfigParser()
    config.read(fname_config)
    geometry = config['GEOMETRY']
    sim = ppp.paraProp(iceDepth=float(geometry['iceDepth']), iceLength=float(geometry['iceLength']),
                       dx=float(geometry['dx']), dz=float(geometry['dz']),
                       airHeight=float(geometry['airHeight']))
    return sim

def create_tx_signal(fname_config):
    '''
    Defines time-domain at transmitter

    Input:
     - fname_config : config file -> sets all the parameters of the simulation
    '''
    config = configparser.ConfigParser()
    config.read(fname_config)
    signal_config = config['TX_SIGNAL']
    sig_tx = tx_signal(amplitude=float(signal_config['amplitude']), frequency=float(signal_config['freq_centre']),
                       bandwidth=float(signal_config['bandwidth']), t_centre=float(signal_config['t_centre']),
                       tmax=float(signal_config['t_max']), dt=float(signal_config['dt']),
                       freqMin=float(signal_config['freqMin']), freqMax=float(signal_config['freqMax']))
    return sig_tx

def create_receiver_array(fname_config):
    '''
        Defines receiver array

        Input:
         - fname_config : config file -> sets all the parameters of the simulation
    '''
    config = configparser.ConfigParser()
    config.read(fname_config)
    receiver_config = config['RECEIVER']
    rx_depths = np.arange(float(receiver_config['minDepth']),
                          float(receiver_config['maxDepth']) + float(receiver_config['dRX_z']),
                          float(receiver_config['dRX_z']))
    rx_ranges = np.arange(float(receiver_config['minRange']),
                          float(receiver_config['maxRange']) + float(receiver_config['dRX_x']),
                          float(receiver_config['dRX_x']))
    return rx_ranges, rx_depths

def create_rx_ranges(fname_config):
    '''
            Defines receiver ranges from an external file

            Input:
             - fname_config : config file -> sets all the parameters of the simulation
    '''
    config = configparser.ConfigParser()
    config.read(fname_config)
    receiver_config = config['RECEIVER']
    fname_rx_ranges = str(receiver_config['fname_rx_ranges'])
    rx_x_list = []
    with open(fname_rx_ranges,'r') as f:
        for line in f:
            cols = line.split()
            rx_x = float(cols[0])
            rx_x_list.append(rx_x)
    rx_ranges = np.array(rx_x_list)
    return rx_ranges

def create_rxList(rx_x, rx_z):
    rxList = []
    for i in range(len(rx_x)):
        for j in range(len(rx_z)):
            rx_ij = rx(rx_x[i], rx_z[j])
            rxList.append(rx_ij)
    return rxList

def create_transmitter_array(fname_config):
    config = configparser.ConfigParser()
    config.read(fname_config)
    transmitter_config = config['TRANSMITTER']
    tx_depths = np.arange(float(transmitter_config['minSource']),
                          float(transmitter_config['maxSource']) + float(transmitter_config['dTX']),
                          float(transmitter_config['dTX']))
    return tx_depths

def create_hdf_bscan(fname, sim, tx_signal, tx_depths, rx_ranges, rx_depths, comment=""):
    '''
    Creates a HDF (fname.h5) file that saves the simulation data -> dimensions, pulse data, receiver configuration
    receiver data

    Inputs:
        - fname: file name (should include path/to/file.h5
        - sim : paraProp object -> includes simulation dimensions
        - tx_signal : tx_signal object -> includes the transmitter pulse
        - tx_depths : includes the depths of the transmitter
        - rx_depths :
    '''
    output_hdf = h5py.File(fname, 'w')
    output_hdf.attrs["iceDepth"] = sim.iceDepth
    output_hdf.attrs["iceLength"] = sim.iceLength
    output_hdf.attrs["airHeight"] = sim.airHeight
    output_hdf.attrs["dx"] = sim.dx
    output_hdf.attrs["dz"] = sim.dz
    output_hdf.create_dataset('n_matrix', data=sim.get_n())

    output_hdf.attrs["Amplitude"] = tx_signal.amplitude
    output_hdf.attrs["freqCentral"] = tx_signal.frequency
    output_hdf.attrs["Bandwidth"] = tx_signal.bandwidth
    output_hdf.attrs["freqMax"] = tx_signal.freqMax
    output_hdf.attrs["freqMin"] = tx_signal.freqMin
    output_hdf.attrs["freqSample"] = tx_signal.fsample
    output_hdf.attrs["freqNyquist"] = tx_signal.freq_nyq
    output_hdf.attrs["tCentral"] = tx_signal.t_centre
    output_hdf.attrs["tSample"] = tx_signal.tmax
    output_hdf.attrs["dt"] = tx_signal.dt
    output_hdf.attrs["nSamples"] = tx_signal.nSamples

    n_profile_data = np.zeros((2, len(sim.get_n(x=0))), dtype='complex')
    n_profile_data[0] = sim.z
    n_profile_data[1] = sim.get_n(x=0)

    nRX_x = len(rx_ranges)
    nRX_z = len(rx_depths)
    rxArray = np.ones((nRX_x, nRX_z, 2))
    for i in range(nRX_x):
        for j in range(nRX_z):
            rxArray[i,j,0] = rx_ranges[i]
            rxArray[i,j,1] = rx_depths[j]

    output_hdf.create_dataset("rxArray", data=rxArray)
    output_hdf.create_dataset('n_profile', data=n_profile_data)
    output_hdf.create_dataset("source_depths", data=tx_depths)
    output_hdf.create_dataset('tspace', data=tx_signal.tspace)
    output_hdf.create_dataset('signalPulse', data=tx_signal.pulse)
    output_hdf.create_dataset('signalSpectrum', data=tx_signal.spectrum)
    output_hdf.create_dataset("rx_range", data= rx_ranges)
    output_hdf.create_dataset("rx_depths", data = rx_depths)

    output_hdf.attrs["comment"] = comment
    return output_hdf

def create_memmap(file, dimensions, data_type ='complex'):
    A = open_memmap(file, shape = dimensions, mode='w+', dtype = data_type)
    return A

class bscan:
    def load_sim(self, fname):
        input_hdf = h5py.File(fname,'r')
        self.fname = fname
        self.iceDepth = float(input_hdf.attrs["iceDepth"])
        self.iceLength = float(input_hdf.attrs["iceLength"])
        self.airHeight = float(input_hdf.attrs["airHeight"])
        self.dx = float(input_hdf.attrs["dx"])
        self.dz = float(input_hdf.attrs["dz"])

        Amplitude = float(input_hdf.attrs["Amplitude"])
        freqCentral = float(input_hdf.attrs["freqCentral"])
        freqMin = float(input_hdf.attrs["freqMin"])
        Bandwidth = float(input_hdf.attrs['Bandwidth'])
        freqMax = float(input_hdf.attrs["freqMax"])
        tCentral = float(input_hdf.attrs["tCentral"])
        tSample = float(input_hdf.attrs["tSample"])
        dt = float(input_hdf.attrs["dt"])

        self.tx_signal = tx_signal(amplitude=Amplitude, frequency=freqCentral, bandwidth=Bandwidth, freqMin=freqMin, freqMax=freqMax, t_centre=tCentral, dt=dt, tmax=tSample)
        self.tx_signal.pulse = np.array(input_hdf.get('signalPulse'))
        self.tx_depths = np.array(input_hdf.get('source_depths'))
        self.rx_depths = np.array(input_hdf.get('rx_depths'))
        self.rx_ranges = np.array(input_hdf.get('rx_range'))

        self.tspace = self.tx_signal.tspace
        self.nSamples = self.tx_signal.nSamples
        self.dt = self.tx_signal.dt

        self.nRx_x = len(self.rx_ranges)
        self.nRX_z = len(self.rx_depths)
        self.nTX = len(self.tx_depths)

        self.n = np.array(input_hdf.get('n_matrix'))
        n_data = np.array(input_hdf.get('n_profile'))
        self.z_profile = n_data[0,:]
        self.n_profile = n_data[1,:]

        self.comment = input_hdf.attrs["comment"]
        self.bscan_sig = np.array(input_hdf.get('bscan_sig'))
        input_hdf.close()
        #TODO: Consider creating a temporary memmap to store bscan -> in case of very large files
        '''
        #check size
        d_size = float(os.path.getsize(fname))
        if d_size < 1e9:
            self.bscan_sig = np.array(input_hdf.get('bscan_sig'))
            self.fname_temp = None
            input_hdf.close()
        else:
            self.fname_temp = fname[:-3] + '_temporary.npy' # temporary memmap
            print('large file size: ', d_size/1e9, ' GB for ', fname, '\n create temporary memmap ' + self.fname_temp)
            self.bscan_sig = create_memmap(self.fname_temp, dimensions=(self.nTX, self.nRx_x, self.nRX_z, self.nSamples))
            bscan_in = np.array(input_hdf.get('bscan_sig'))
            for i in range(self.nTX):
                for j in range(self.nRx_x):
                    for k in range(self.nRX_z):
                        self.bscan_sig[i,j,k,:] = bscan_in[i,j,k,:]
            input_hdf.close()
            bscan_in = None
        '''

    def get_ascan(self, zTx, xRx, zRx):
        ii_tx = util.findNearest(self.tx_depths, zTx)
        ii_rx_x = util.findNearest(self.rx_ranges, xRx)
        ii_rx_z = util.findNearest(self.rx_depths, zRx)
        self.ascan = self.bscan_sig[ii_tx, ii_rx_x, ii_rx_z, :]
        return self.ascan

    def bscan_parallel(self, xRx):
        self.bscan_plot = np.zeros((self.nTX, self.tx_signal.nSamples), dtype='complex')
        ii_rx_x = util.findNearest(self.rx_ranges, xRx)
        for i in range(self.nTX):
            self.bscan_plot[i] = self.bscan_sig[i, ii_rx_x, i, :]
        return self.bscan_plot

    def bscan_tx_fixed(self, zTx, xRx):
        self.bscan_plot = np.zeros((self.nTX, self.tx_signal.nSamples), dtype='complex')
        ii_rx_x = util.findNearest(self.rx_ranges, xRx)
        ii_tx = util.findNearest(self.tx_depths, zTx)

        for i in range(self.nRX_z):
            self.bscan_plot[i] = self.bscan_sig[ii_tx, ii_rx_x, i, :]
        return self.bscan_plot

    def bscan_rx_fixed(self, xRx, zRx):
        self.bscan_plot = np.zeros((self.nTX, self.tx_signal.nSamples), dtype='complex')
        ii_rx_x = util.findNearest(self.rx_ranges, xRx)
        ii_rx_z = util.findNearest(self.rx_depths, zRx)

        for i in range(self.nRX_z):
            self.bscan_plot[i] = self.bscan_sig[i, ii_rx_x, ii_rx_z, :]
        return self.bscan_plot

    def bscan_depth_fixed(self, zDepth):
        self.bscan_plot = np.zeros((self.nRx_x, self.tx_signal.nSamples), dtype='complex')
        ii_z = util.findNearest(self.tx_depths, zDepth)

        for i in range(self.nRx_x):
            self.bscan_plot[i] = self.bscan_sig[ii_z, i, ii_z, :]
        return self.bscan_plot