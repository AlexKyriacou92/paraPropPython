import util
import math as m
import numpy as np
from inspect import signature
from scipy.interpolate import interp1d
from scipy import signal
# A. Kyriacou

'''
This module defines the transmitter and transmitted signal
Trasnmitter acts as the source for the radio emission that is observed at the receiver points
'''

class tx_signal:
    def __init__(self, frequency, bandwidth, t_centre, dt, tmax, amplitude = 1, freqMin=None, freqMax=None):
        self.amplitude = amplitude
        self.frequency = frequency
        self.bandwidth = bandwidth
        self.t_centre = t_centre
        self.dt = dt
        self.fsample = 1/dt
        self.freq_nyq = 1/(2*dt)
        self.tmax = tmax
        self.nSamples = int(tmax/dt)
        self.tspace = np.linspace(0, tmax, self.nSamples)
        self.freq_space = np.fft.fftfreq(self.nSamples, self.dt)

        if freqMin == None:
            self.freqMin = 0
        else:
            self.freqMin = freqMin
        if freqMax == None:
            self.freqMax = self.freq_nyq
        else:
            self.freqMax = freqMax

        '''
        if freqMax == None:
            self.freqMax = self.frequency + self.bandwidth/2
        else:
            self.freqMax = freqMax
        if freqMin == None:
            self.freqMin = self.frequency - self.bandwidth/2
        else:
            self.freqMin = freqMin
        '''

    def get_gausspulse_complex(self, suppression = -60):
        frac_bandwidth = self.bandwidth / self.frequency
        pulse_r = self.amplitude * signal.gausspulse(self.tspace - self.t_centre, fc=self.frequency, bw=frac_bandwidth, bwr=suppression)
        pulse_i = self.amplitude * np.array(signal.gausspulse(self.tspace - self.t_centre, fc=self.frequency, bw=frac_bandwidth, bwr=suppression, retquad=True))[1]
        self.pulse = pulse_r + 1j*pulse_i
        self.spectrum = np.fft.fft(self.pulse)
        return self.pulse

    def get_gausspulse(self, suppression = -60):
        frac_bandwidth = self.bandwidth / self.frequency
        pulse_r = self.amplitude * signal.gausspulse(self.tspace - self.t_centre, fc=self.frequency, bw=frac_bandwidth, bwr=suppression)
        self.pulse = pulse_r
        self.spectrum = np.fft.fft(self.pulse)
        return self.pulse

    def get_gausspulse_real(self, suppression = -60):
        frac_bandwidth = self.bandwidth / self.frequency
        pulse_r = self.amplitude * signal.gausspulse(self.tspace - self.t_centre, fc=self.frequency, bw=frac_bandwidth, bwr=suppression)
        self.pulse = pulse_r
        self.spectrum = np.fft.fft(self.pulse)
        return self.pulse

    def get_spectrum(self): #NOTE: pulse must be defined before
        return self.spectrum

'''
class transmitter:
    def __init__(self, z, xO = 0):
        self.xO = xO
        self.z = z
'''