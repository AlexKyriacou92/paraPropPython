import h5py
import paraPropPython as ppp
from transmitter import tx_signal
from receiver import receiver as rx
import numpy as np
from numpy.lib.format import open_memmap
import peakutils as pku
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
    if 'refDepth' in geometry:
        sim = ppp.paraProp(iceDepth=float(geometry['iceDepth']), iceLength=float(geometry['iceLength']),
                           dx=float(geometry['dx']), dz=float(geometry['dz']), airHeight=float(geometry['airHeight']),
                           refDepth=float(geometry['refDepth']))
    else:
        sim = ppp.paraProp(iceDepth=float(geometry['iceDepth']), iceLength=float(geometry['iceLength']),
                           dx=float(geometry['dx']), dz=float(geometry['dz']),
                           airHeight=float(geometry['airHeight']))
    return sim

def get_IR_from_config(fname_config, antenna='TX'):
    config = configparser.ConfigParser()
    config.read(fname_config)
    if antenna == 'TX':
        signal_config = config['TX_SIGNAL']
        fname_IR = signal_config['fname_IR']
        IR, IR_freq = util.get_IR_from_file(fname_IR)
    elif antenna == 'RX':
        receiver_config = config['RECEIVER']
        fname_IR = receiver_config['fname_IR']
        IR, IR_freq = util.get_IR_from_file(fname_IR)
    return IR, IR_freq

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

    if config.has_option('TX_SIGNAL', 'noise') == True:
        sig_tx = tx_signal(amplitude=float(signal_config['amplitude']), frequency=float(signal_config['freq_centre']),
                  bandwidth=float(signal_config['bandwidth']), t_centre=float(signal_config['t_centre']),
                  tmax=float(signal_config['t_max']), dt=float(signal_config['dt']),
                  freqMin=float(signal_config['freqMin']), freqMax=float(signal_config['freqMax']),
                  noise_amplitude=float(config['RECEIVER']['noise']))
    return sig_tx

def create_tx_signal_from_file(fname_config):
    config = configparser.ConfigParser()
    config.read(fname_config)
    signal_config = config['TX_SIGNAL']
    fname_tx = signal_config['fname_tx_signal']

    tspace_list = []
    tx_sig_pulse_list = []
    with open(fname_tx) as fin:
        for line in fin:
            cols = line.split()
            ti = float(cols[0])
            tx_sig_i = float(cols[1])
            tspace_list.append(ti)
            tx_sig_pulse_list.append(tx_sig_i)
    tspace = np.array(tspace_list)
    tx_pulse = np.array(tx_sig_pulse_list)
    sig_tx = tx_signal(amplitude=float(signal_config['amplitude']), frequency=float(signal_config['freq_centre']),
                  bandwidth=float(signal_config['bandwidth']), t_centre=float(signal_config['t_centre']),
                  tmax=float(signal_config['t_max']), dt=float(signal_config['dt']),
                  freqMin=float(signal_config['freqMin']), freqMax=float(signal_config['freqMax']),
                  noise_amplitude=float(config['RECEIVER']['noise']))
    sig_tx.set_pulse(tx_pulse, tspace)
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

def create_rxList_from_file(fname_config):
    config = configparser.ConfigParser()
    config.read(fname_config)
    receiver_config = config['RECEIVER']
    fname_rx_ranges = str(receiver_config['fname_receivers'])
    rxList = []

    f_recievers = open(fname_rx_ranges, 'r')
    next(f_recievers)
    if config.has_option('RECEIVER', 'noise') == True:
        noise_amplitude = float(receiver_config['noise'])
        for line in f_recievers:
            cols = line.split()
            rx_x = float(cols[0])
            rx_z = float(cols[1])
            rx_i = rx(x=rx_x, z=rx_z, noise_amplitude=noise_amplitude)
            rxList.append(rx_i)
    else:
        for line in f_recievers:
            cols = line.split()
            rx_x = float(cols[0])
            rx_z = float(cols[1])
            rx_i = rx(x=rx_x, z=rx_z)
            rxList.append(rx_i)
    return rxList

def create_transmitter_array(fname_config):
    config = configparser.ConfigParser()
    config.read(fname_config)
    transmitter_config = config['TRANSMITTER']
    fname_tx_depths = str(transmitter_config['fname_transmitters'])
    tx_depths = []
    with open(fname_tx_depths, 'r') as f_in:
        for line in f_in:
            cols = line.split()
            tx_depths.append(float(cols[0]))
    tx_depths = np.array(tx_depths)
    return tx_depths

def create_transmitter_array_from_file(fname_config):
    config = configparser.ConfigParser()
    config.read(fname_config)
    transmitter_config = config['TRANSMITTER']
    fname_tx = transmitter_config['fname_transmitters']
    tx_depths = []
    with open(fname_tx, 'r') as fin:
        for line in fin:
            cols = line.split()
            z_tx = float(cols[0])
            tx_depths.append(z_tx)
    tx_depths = np.array(tx_depths)
    return tx_depths

def create_ascan_hdf(fname_config, tx_signal, nprof_data, zprof_data, fname_output):
    sim = create_sim(fname_config)
    rxList = create_rxList_from_file(fname_config)
    tx_depths = create_transmitter_array(fname_config)

    output_hdf = h5py.File(fname_output, 'w')
    output_hdf.attrs["iceDepth"] = sim.iceDepth
    output_hdf.attrs["iceLength"] = sim.iceLength
    output_hdf.attrs["airHeight"] = sim.airHeight
    output_hdf.attrs["dx"] = sim.dx
    output_hdf.attrs["dz"] = sim.dz

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

    output_hdf.create_dataset('n_profile', data=nprof_data)
    output_hdf.create_dataset('z_profile', data=zprof_data)
    output_hdf.create_dataset("source_depths", data=tx_depths)
    output_hdf.create_dataset('tspace', data=tx_signal.tspace)
    output_hdf.create_dataset('signalPulse', data=tx_signal.pulse)
    output_hdf.create_dataset('signalSpectrum', data=tx_signal.spectrum)

    rxList_positions = np.ones((len(rxList), 2))
    for i in range(len(rxList)):
        rx_i = rxList[i]
        rxList_positions[i, 0] = rx_i.x
        rxList_positions[i, 1] = rx_i.z

    output_hdf.create_dataset("rxList", data=rxList_positions)
    return output_hdf

class ascan:
    def load_from_hdf(self, fname_hdf):
        input_hdf = h5py.File(fname_hdf, 'r')
        self.fname = fname_hdf
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

        self.tx_signal = tx_signal(amplitude=Amplitude, frequency=freqCentral, bandwidth=Bandwidth, freqMin=freqMin,
                                   freqMax=freqMax, t_centre=tCentral, dt=dt, tmax=tSample)
        self.tx_signal.pulse = np.array(input_hdf.get('signalPulse'))
        self.tx_signal.spectrum = np.array(input_hdf.get('signalSpectrum'))
        self.tx_depths = np.array(input_hdf.get('source_depths'))

        rxList_positions = np.array(input_hdf.get('rxList'))
        rxList = []
        for i in range(len(rxList_positions)):
            rx_i = rx(x=rxList_positions[i,0], z= rxList_positions[i,1])
            rxList.append(rx_i)
        self.rxList = rxList
        self.tspace = self.tx_signal.tspace
        self.nSamples = self.tx_signal.nSamples
        self.dt = self.tx_signal.dt

        self.nRX = len(self.rxList)
        self.nTX = len(self.tx_depths)
        self.n_profile = np.array(input_hdf.get('n_profile')) # 2 x nZ array -> includes sim.z and sim.n(x=0,z)
        self.z_profile = np.array(input_hdf.get('z_profile'))

        if 'rxSpectrum' in input_hdf.keys():
            self.spectrum_array = np.array(input_hdf.get('rxSpectrum'))
        if 'rxSignal' in input_hdf.keys():
            self.ascan_array = np.array(input_hdf.get('rxSignal'))
        input_hdf.close()

    def save_spectrum(self, fname_npy):
        spectrum = np.load(fname_npy, 'r')
        self.spectrum_array = spectrum
        self.ascan_array = np.zeros((self.nTX, self.nRX, self.nSamples),dtype='complex')
        for i in range(self.nTX):
            for j in range(self.nRX):
                spectrum_ij = self.spectrum_array[i,j]
                #spectrum_ishift = np.fft.ifftshift(spectrum_ij)
                spectrum_ij_flip = np.flip(spectrum_ij)
                ascan_ij = np.fft.ifft(spectrum_ij_flip)
                self.ascan_array[i,j] = ascan_ij
        with h5py.File(self.fname, 'r+') as hdf_in:
            hdf_in.create_dataset('rxSpectrum', data=self.spectrum_array)
            hdf_in.create_dataset('rxSignal', data=self.ascan_array)

    def load_spectrum(self, spectrum):
        self.spectrum_array = spectrum
        self.ascan_array = np.zeros((self.nTX, self.nRX, self.nSamples),dtype='complex')
        for i in range(self.nTX):
            for j in range(self.nRX):
                spectrum_ij = self.spectrum_array[i,j]
                #spectrum_ishift = np.fft.ifftshift(spectrum_ij)
                #spectrum_ij_flip = np.flip(spectrum_ij)
                #ascan_ij = np.fft.ifft(spectrum_ij_flip)
                ascan_ij = np.fft.ifft(np.flip(spectrum_ij))
                self.ascan_array[i,j] = ascan_ij
        return self.ascan_array
    def get_ascan(self, z_tx, x_rx, z_rx):
        ii_tx = util.findNearest(self.tx_depths, z_tx)
        r_list = []
        for i in range(self.nRX):
            x_rx_i = self.rxList[i].x
            z_rx_i = self.rxList[i].z
            r_sq = (x_rx_i - x_rx)**2 + (z_rx_i - z_rx)**2
            r_abs = np.sqrt(r_sq)
            r_list.append(r_abs)
        jj_rx = np.argmin(r_list)
        ascan_ij = self.ascan_array[ii_tx, jj_rx]
        return ascan_ij
    def get_spectrum(self, z_tx, x_rx, z_rx):
        ii_tx = util.findNearest(self.tx_depths, z_tx)
        r_list = []
        for i in range(self.nRX):
            x_rx_i = self.rxList[i].x
            z_rx_i = self.rxList[i].z
            r_sq = (x_rx_i - x_rx)**2 + (z_rx_i - z_rx)**2
            r_abs = np.sqrt(r_sq)
            r_list.append(r_abs)
        jj_rx = np.argmin(r_list)
        spectrum_ij = self.spectrum_array[ii_tx, jj_rx]
        return spectrum_ij
def save_field_to_file(sim, fname_out, mode = '1D'):
    field_complex = sim.get_field()
    output_hdf = h5py.File(fname_out, 'w')
    output_hdf.attrs["iceDepth"] = sim.iceDepth
    output_hdf.attrs["iceLength"] = sim.iceLength
    output_hdf.attrs["airHeight"] = sim.airHeight
    output_hdf.attrs["dx"] = sim.dx
    output_hdf.attrs["dz"] = sim.dz
    output_hdf.attrs["centerFreq"] = sim.centerFreq
    output_hdf.create_dataset('source', data=sim.source)
    output_hdf.create_dataset('field', data=field_complex)
    if mode == '1D':
        n_profile = sim.get_n(x=0)
        output_hdf.create_dataset('n_profile', data=n_profile.real)
        output_hdf.create_dataset('z_profile', data=sim.z)
    elif mode == '2D':
        n_profile = sim.get_n()
        output_hdf.create_dataset('n_profile', data=n_profile.real)
        output_hdf.create_dataset('z_profile', data=sim.z)
    output_hdf.close()

def get_field_from_file(fname_field):
    with h5py.File(fname_field,'r') as input_hdf:
        iceDepth = float(input_hdf.attrs["iceDepth"])
        iceLength = float(input_hdf.attrs["iceLength"])
        airHeight = float(input_hdf.attrs["airHeight"])
        dx = float(input_hdf.attrs["dx"])
        dz = float(input_hdf.attrs["dz"])

        #Open Simulation
        sim = ppp.paraProp(iceLength=iceLength, iceDepth=iceDepth, dx=dx, dz=dz, airHeight=airHeight)

        #Save Field from Input File
        field_out = np.array(input_hdf.get('field'))
        sim.source = np.array(input_hdf.get('source'))
        #n_profile = np.array(input_hdf.get('n_profile'))
        #z_profile = np.array(input_hdf.get('z_profile'))
        #sim.set_n(nVec=n_profile, zVec=z_profile)
        sim.field = field_out
        sim.centerFreq = float(input_hdf.attrs['centerFreq'])
    return sim

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

def create_hdf_FT(fname, sim, tx_signal, tx_depths, rxList, comment=""):
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

    output_hdf.create_dataset('n_profile', data=n_profile_data)
    output_hdf.create_dataset("source_depths", data=tx_depths)
    output_hdf.create_dataset('tspace', data=tx_signal.tspace)
    output_hdf.create_dataset('signalPulse', data=tx_signal.pulse)
    output_hdf.create_dataset('signalSpectrum', data=tx_signal.spectrum)

    rxList_positions = np.ones((len(rxList), 2))
    for i in range(len(rxList)):
        rx_i = rxList[i]
        rxList_positions[i, 0] = rx_i.x
        rxList_positions[i, 1] = rx_i.z

    output_hdf.create_dataset("rxList", data=rxList_positions)
    output_hdf.attrs["comment"] = comment

    return output_hdf

def create_memmap(file, dimensions, data_type ='complex'):
    A = open_memmap(file, shape = dimensions, mode='w+', dtype = data_type)
    return A

class bscan_rxList: #This one is a nTx x nRx dimension bscan
    def setup_from_config(self, fname_config, n_profile = None, z_profile = None, bscan_npy = None):
        self.sim = create_sim(fname_config)
        self.rxList = create_rxList_from_file(fname_config)
        self.tx_signal = create_tx_signal(fname_config)
        self.tx_depths = create_transmitter_array(fname_config)
        #if n_profile != None:
        self.n_profile = n_profile
        #if z_profile != None:
        self.z_profile = z_profile
        #if bscan_npy != None:
        self.bscan_sig = bscan_npy

        self.iceDepth = self.sim.iceDepth
        self.iceLength = self.sim.iceLength
        self.airHeight = self.sim.airHeight
        self.dx = self.sim.dx
        self.dz = self.sim.dz

        self.tspace = self.tx_signal.tspace
        self.nSamples = self.tx_signal.nSamples
        self.dt = self.tx_signal.dt

        self.nRX = len(self.rxList)
        self.nTX = len(self.tx_depths)

    def set_nprofile(self, n_profile, z_profile):
        self.n_profile = n_profile
        self.z_profile = z_profile

    def set_bscan_sig(self, bscan_npy):
        self.bscan_sig = bscan_npy

    def load_sim(self, fname):
        input_hdf = h5py.File(fname, 'r')
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

        self.tx_signal = tx_signal(amplitude=Amplitude, frequency=freqCentral, bandwidth=Bandwidth, freqMin=freqMin,
                                   freqMax=freqMax, t_centre=tCentral, dt=dt, tmax=tSample)
        self.tx_signal.pulse = np.array(input_hdf.get('signalPulse'))
        self.tx_depths = np.array(input_hdf.get('source_depths'))

        rxList_positions = np.array(input_hdf.get('rxList'))
        rxList = []
        for i in range(len(rxList_positions)):
            rx_i = rx(x=rxList_positions[i,0], z= rxList_positions[i,1])
            rxList.append(rx_i)
        self.rxList = rxList
        self.tspace = self.tx_signal.tspace
        self.nSamples = self.tx_signal.nSamples
        self.dt = self.tx_signal.dt

        self.nRX = len(self.rxList)
        self.nTX = len(self.tx_depths)

        self.n = np.array(input_hdf.get('n_matrix')) # 2D Ref-index matrix
        n_data = np.array(input_hdf.get('n_profile')) # 2 x nZ array -> includes sim.z and sim.n(x=0,z)
        self.z_profile = n_data[0, :]
        self.n_profile = n_data[1, :]

        self.comment = input_hdf.attrs["comment"]
        self.bscan_sig = np.array(input_hdf.get('bscan_sig'))
        input_hdf.close()

    def get_ascan(self, txNum, rxNum):
        ascan = self.bscan_sig[txNum, rxNum]
        return ascan

    def get_peak(self, txNum, rxNum, mode='Max', thres0=0.1):
        ascan = abs(self.bscan_sig[txNum, rxNum])
        if mode == 'Max':
            ii_max = np.argmax(ascan)
            tpeak = self.tspace[ii_max]
            amp_peak = ascan[ii_max]
        elif mode == 'First':
            ii_peaks = pku.indexes(ascan, thres=thres0)
            ii_first = ii_peaks[0]
            tpeak = self.tspace[ii_first]
            amp_peak = ascan[ii_first]
        elif mode == 'All':
            ii_peaks = pku.indexes(ascan, thres=thres0)
            nPeaks = len(ii_peaks)
            tPeaks = np.zeros(nPeaks)
            AmpPeaks = np.zeros(nPeaks)
            for i in range(nPeaks):
                tPeaks[i] = self.tspace[ii_peaks[i]]
                AmpPeaks[i] = ascan[ii_peaks[i]]
            return tPeaks, AmpPeaks
        else:
            ii_max = np.argmax(ascan)
            tpeak = self.tspace[ii_max]
            amp_peak = ascan[ii_max]
        return tpeak, amp_peak

    def get_ascan_from_depth(self, z_tx, x_rx, z_rx):
        txNum = util.findNearest(self.tx_depths, z_tx)
        rxNum = 0
        rxList = self.rxList
        range_tot = []
        nRx = len(rxList)
        for i in range(nRx):
            ri = np.sqrt((x_rx - self.rxList[i].x)**2 + (z_rx - self.rxList[i].z)**2)
            range_tot.append(ri)
        rxNum = util.findNearest(range_tot, 0)
        #rx_select = rxList[rxNum]
        #print(rx_select.x, rx_select.z)
        ascan = self.bscan_sig[txNum, rxNum]
        return ascan

    def get_actual_position(self, z_tx, x_rx, z_rx):
        txNum = util.findNearest(self.tx_depths, z_tx)
        rxList = self.rxList

        nRx = len(rxList)
        range_tot = []
        for i in range(nRx):
            ri = np.sqrt(abs(x_rx - self.rxList[i].x)**2 + abs(z_rx - self.rxList[i].z)**2)
            range_tot.append(ri)
        rxNum = util.findNearest(range_tot, 0)
        rx_select = rxList[rxNum]
        return self.tx_depths[txNum], rx_select.x, rx_select.z

    def get_spectrum_from_depth(self, z_tx, x_rx, z_rx):
        txNum = util.findNearest(self.tx_depths, z_tx)
        rxNum = 0
        rxList = self.rxList
        nRX = self.nRX
        for i in range(nRX):
            rx_i = self.rxList[i]
            if rx_i.x == x_rx and rx_i.z == z_rx:
                rxNum = i
        rx_select = rxList[rxNum]
        #print(rx_select.x, rx_select.z)
        ascan = self.bscan_sig[txNum, rxNum]
        ascan_fft = np.fft.rfft(ascan)
        ascan_fft /= float(self.nSamples)
        freq_space = np.fft.rfftfreq(self.nSamples, self.dt)
        return ascan_fft, freq_space

    def bscan_parallel(self, xRx):
        self.bscan_plot = np.zeros((self.nTX, self.tx_signal.nSamples), dtype='complex')
        #ii_rx_x = util.findNearest(self.rx_ranges, xRx)
        for i in range(self.nTX):
            self.bscan_plot[i] = self.bscan_sig[i, i, :]
        return self.bscan_plot

    def bscan_parallel2(self, xRx):
        nSamples = len(self.bscan_sig[0,0,:])
        self.bscan_plot = np.zeros((self.nTX, nSamples), dtype='complex')
        for i in range(self.nTX):
            self.bscan_plot[i] = self.bscan_sig[i, i, :]
        return self.bscan_plot
    def bscan_rx_fixed(self, zRx):
        self.bscan_plot = np.zeros((self.nTX, self.tx_signal.nSamples), dtype='complex')
        ii_rx_z = util.findNearest(self.tx_depths, zRx)

        for i in range(self.nTX):
            self.bscan_plot[i] = self.bscan_sig[i, ii_rx_z, :]
        return self.bscan_plot

    def bscan_tx_fixed(self, zTx):
        self.bscan_plot = np.zeros((self.nTX, self.tx_signal.nSamples), dtype='complex')
        ii_z = util.findNearest(self.tx_depths, zTx)

        for i in range(self.nTX):
            self.bscan_plot[i] = self.bscan_sig[ii_z, i, :]
        return self.bscan_plot

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

class bscan_FT:
    def load_from_hdf(self, fname):
        hdf_data = h5py.File(fname,'r')
        self.fname = fname
        self.fftArray = np.array(hdf_data['fftArray'])
        self.freqList = np.array(hdf_data['freqList']) / 1e9 #Note FT Data is defined in s/Hz, while paraProp uses ns/GHz
        self.rxDepths = np.array(hdf_data['rxDepths'])
        self.rxRanges = np.array(hdf_data['rxRanges'])
        self.tspace = np.array(hdf_data['tspace']) * 1e9 #Note FT Data is defined in s/Hz, while paraProp uses ns/GHz
        self.txDepths = np.array(hdf_data['txDepths'])
        self.dt = abs(self.tspace[1] - self.tspace[0])
        self.nSamples = len(self.fftArray[0])
        self.nData = len(self.fftArray)
        hdf_data.close()

    def get_ascan(self, z_tx, x_rx, z_rx, tol=0.05):
        ascan_data = None
        for i in range(self.nData):
            z_tx_i = self.txDepths[i]
            x_rx_i = self.rxRanges[i]
            z_rx_i = self.rxDepths[i]

            dz_tx = abs(z_tx_i - z_tx)
            dx_rx = abs(x_rx_i - x_rx)
            dz_rx = abs(z_rx_i - z_rx)
            if dz_tx < tol and dx_rx < tol and dz_rx < tol:
                ascan_data = self.fftArray[i]
                break
        '''
        if ascan_data == None:
            print('error, no matrching data found')
        else:
            print('match found')
        '''
        return ascan_data
