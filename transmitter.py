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
    def __init__(self, frequency, bandwidth, t_centre, dt, tmax, amplitude = 1, noise_amplitude = 0, freqMin=None, freqMax=None):
        self.amplitude = amplitude
        self.frequency = frequency # Central Frequency
        self.bandwidth = bandwidth
        self.t_centre = t_centre
        self.dt = dt
        self.fsample = 1/dt
        self.freq_nyq = 1/(2*dt)
        self.tmax = tmax
        self.nSamples = int(tmax/dt)
        self.tspace = np.linspace(0, tmax, self.nSamples)
        self.freq_space = np.fft.fftfreq(self.nSamples, self.dt)
        self.noise_amplitude = noise_amplitude

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
    
    def set_pulse(self, pulse_data, tspace_pulse):
        self.pulse = pulse_data
        self.spectrum = np.fft.fft(self.pulse)
        self.dt = abs(tspace_pulse[1] - tspace_pulse[0])
        self.fsample = 1/self.dt
        self.freq_nyq = 1/(2*self.dt)
        self.tmax = max(tspace_pulse)
        self.nSamples = len(pulse_data)
        self.tspace = tspace_pulse
        self.freq_space = np.fft.fftfreq(self.nSamples, self.dt)

    def do_impulse_response(self, IR, IR_freq):
        spectrum_shift = np.fft.fftshift(self.spectrum)
        freq_shift = np.fft.fftshift(self.freq_space)
        nFreq = len(freq_shift)
        IR_fmin = min(IR_freq)
        IR_fmax = max(IR_freq)
        self.IR = np.ones(nFreq)
        for i in range(nFreq):
            freq_i = self.freq_space[i]
            if freq_i < IR_fmin:
                self.IR[i] = 0
            elif freq_i > IR_fmax:
                self.IR[i] = 0
            elif freq_i <= IR_fmax and freq_i >= IR_fmin:
                jj = util.findNearest(IR_freq, freq_i)
                self.IR[i] = IR[jj]

        spectrum_shift *= self.IR
        self.spectrum = np.fft.ifftshift(spectrum_shift)
        self.pulse = np.fft.ifft(self.spectrum)

        return self.pulse


    def add_gaussian_noise(self):
        nSamples = len(self.spectrum)
        noise_amplitude = self.noise_amplitude
        if noise_amplitude > 0:
            self.noise = noise_amplitude*np.random.normal(0, noise_amplitude, nSamples)
            self.spectrum += self.noise
            self.pulse = np.fft.ifft(self.spectrum)

    #TODO -> Add Data Defined Noise Spectrum
    #TODO: White Noise?


'''
class transmitter:
    def __init__(self, z, xO = 0):
        self.xO = xO
        self.z = z
'''

# ==========================================

''''
def set_pulse(self, sigVec):
    if len(sigVec) == self.nSamples:
        self.pulse = sigVec
        return self.pulse
    else:
        print('error, signal vector must have the same dimensions, pulse is set to zero')
        self.pulse = np.zeros(self.nSamples)
        return self.pulse
'''