# paraPropPython
# c. sbrocco, s. prohira
# A. Kyriacou

import util
import math as m
import numpy as np
from inspect import signature
from scipy.interpolate import interp1d
from scipy import signal

'''
This module defines the receiver class
    -Receivers (RX) are points in the ice geometry which measure the radio signal from the transmitter
    i.e. the 'observers'
    -In FD mode, RXs are optional -> this will just tell you the amplitude at the RX point
    -In TD mode, RXs are mandatory -> the received signal is reconstructed by the spectrum at the RX
    
    Parameters:
    x -> the receiver range
    z -> the receiver depth (note that z > 0 means below surface, and z < 0 means above)
'''

#TODO: Define Rx array in this file?

class receiver:
    """
    Parameters
    ----------
    x : float
        x position (m)
    z : float
        z position (m)
    """

    def __init__(self, x, z, IR_freq = [], IR_data=[]):
        self.x = x
        self.z = z
        self.IR_data = IR_data
        self.IR_freq = IR_freq

    def setup(self, freq, dt):
        """
        further setup of receiver using simulation parameters

        Parameters
        ----------
        freq : float array
            frequencies (GHz)
        dt : float
            time step (ns)
        """
        self.freq = freq
        self.nFreq = len(freq)
        self.spectrum = np.zeros(self.nFreq, dtype='complex')
        self.spectrum_plus = np.zeros(self.nFreq, dtype='complex')
        self.spectrum_minus = np.zeros(self.nFreq, dtype='complex')
        self.time = np.arange(0, dt * self.nFreq, dt)
        self.IR = np.ones(self.nFreq)
        if len(self.IR_data) > 0 and len(self.IR_freq) > 0:
            IR_fmin = min(self.IR_freq)
            IR_fmax = max(self.IR_freq)
            for i in range(self.nFreq):
                freq_i = self.freq[i]
                if freq_i < IR_fmin:
                    self.IR[i] = 0
                elif freq_i > IR_fmax:
                    self.IR[i] = 0
                elif freq_i <= IR_fmax and freq_i >= IR_fmin:
                    jj = util.findNearest(self.IR_freq, freq_i)
                    self.IR[i] = self.IR_data[jj]
        elif len(self.IR_data) > 0 and len(self.IR_freq) == 0:
            print('error! must defined frequency vector so IR values correspond to to frequencies')
        elif len(self.IR_freq) > 0 and len(self.IR_data) == 0:
            print('warning! you set IR frequnecy values but not the IR data -> IR not implemented')
        #TODO -> Add elif and else to self.IR_data
    def add_spectrum_component(self, f, A):
        """
        adds the contribution of a frequency to the received signal spectrum

        Parameters
        ----------
        f : float
            corresponding frequencie (GHz)
        A : complex float
            complex amplitude of received siganl (V/m???)
        """
        i = util.findNearest(self.freq, f)
        #self.spectrum[i] = A * self.IR[i]
        self.spectrum[i] = A * self.IR[i]
    def add_spectrum_component_minus(self, f, A):
        """
                adds the contribution of a frequency to the received signal spectrum (for the reflected 'minus' signal

                Parameters
                ----------
                f : float
                    corresponding frequencie (GHz)
                A : complex float
                    complex amplitude of received siganl (V/m???)
                """
        i = util.findNearest(self.freq, f)
        self.spectrum_minus[i] = A

    def add_spectrum_component_plus(self, f, A):
        """
                adds the contribution of a frequency to the received signal spectrum (for the reflected 'minus' signal

                Parameters
                ----------
                f : float
                    corresponding frequencie (GHz)
                A : complex float
                    complex amplitude of received siganl (V/m???)
                """
        i = util.findNearest(self.freq, f)
        self.spectrum_plus[i] = A

    def get_spectrum(self):
        """
        gets received signal spectrum

        Returns
        -------
        1-d comlplex float array
        """
        return self.spectrum[:int(len(self.freq) / 2)]

    def get_signal(self):
        """
        gets received signal

        Returns
        -------
        1-d comlplex float array
        """
        return np.flip(util.doIFFT(self.spectrum))

    def get_signal_plus(self):
        '''
        get forward going (positive) signal

        Returns
        --------
        1-d complex float array
        '''
        return np.flip(util.doIFFT(self.spectrum_plus))

    def get_signal_minus(self):
        '''
         get backward going (minus) signal

        Returns
        --------
        1-d complex float array
        '''
        return np.flip(util.doIFFT(self.spectrum_minus))

    def get_frequency(self):
        """
        gets frequency array

        Returns
        -------
        1-d float array
        """
        return abs(self.freq)[:int(len(self.freq) / 2)]

    def get_time(self):
        """
        gets time array

        Returns
        -------
        1-d float array
        """
        return self.time

    def do_impulse_response(self, h):
        '''
        Applies Convolution of received amplitude and antenna's impulse response function
        -> Simply applies
        '''
        self.impulse_response = h
        self.spectrum *= self.impulse_response
        self.spectrum_plus *= self.impulse_response
        self.spectrum_minus *= self.impulse_response

    def get_impulse_response(self):
        return self.impulse_response

#TODO: Add antenna pattern to this module
