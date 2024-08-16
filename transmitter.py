import util
import math as m
import numpy as np
from inspect import signature
from scipy.interpolate import interp1d
from scipy import signal
from scipy.signal.windows import tukey
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
        self.df = 1./(self.dt * self.nSamples)
        self.tspace = np.linspace(0, tmax, self.nSamples)
        self.freq_space = np.fft.fftfreq(self.nSamples, self.dt)
        self.noise_amplitude = noise_amplitude

        self.freq_plus = np.arange(0, 1/self.dt, self.df) #Positive frequency space (0 to 2 f_nyq)
        self.nHalf = int(self.nSamples / 2) + self.nSamples % 2
        print('nHalf = ', self.nHalf)
        self.spectrum_plus = np.zeros(self.nSamples, dtype='complex')

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

    def set_impulse(self):
        jj = util.findNearest(self.tspace, self.t_centre)
        self.pulse = np.zeros(self.nSamples)
        self.pulse[jj] = self.amplitude
        self.spectrum = np.fft.fft(self.pulse)
        self.spectrum_plus = util.doFFT(np.flip(self.pulse))
        return self.pulse

    def get_spectrum_plus(self):
        self.spectrum_plus = util.doFFT(np.flip(self.pulse))
        return self.spectrum_plus

    def get_gausspulse_complex(self, suppression = -60):
        frac_bandwidth = self.bandwidth / self.frequency
        pulse_r = self.amplitude * signal.gausspulse(self.tspace - self.t_centre, fc=self.frequency, bw=frac_bandwidth, bwr=suppression)
        pulse_i = self.amplitude * np.array(signal.gausspulse(self.tspace - self.t_centre, fc=self.frequency, bw=frac_bandwidth, bwr=suppression, retquad=True))[1]
        self.pulse = pulse_r + 1j*pulse_i
        self.spectrum = np.fft.fft(self.pulse)
        self.spectrum_plus = util.doFFT(np.flip(self.pulse))
        return self.pulse

    def get_gausspulse(self, suppression = -60):
        frac_bandwidth = self.bandwidth / self.frequency
        pulse_r = self.amplitude * signal.gausspulse(self.tspace - self.t_centre, fc=self.frequency, bw=frac_bandwidth, bwr=suppression)
        self.pulse = pulse_r
        self.spectrum = np.fft.fft(self.pulse)
        self.spectrum_plus = util.doFFT(np.flip(self.pulse))
        return self.pulse

    def get_gausspulse_real(self, suppression = -60):
        frac_bandwidth = self.bandwidth / self.frequency
        pulse_r = self.amplitude * signal.gausspulse(self.tspace - self.t_centre, fc=self.frequency, bw=frac_bandwidth, bwr=suppression)
        self.pulse = pulse_r
        self.spectrum = np.fft.fft(self.pulse)
        self.spectrum_plus = util.doFFT(np.flip(self.pulse))
        return self.pulse

    def get_spectrum(self): #NOTE: pulse must be defined before
        return np.fft.fftshift(self.spectrum)

    def get_freq_space(self):
        return np.fft.fftshift(self.freq_space)

    def apply_bandpass(self, fmin, fmax, order=3):
        self.pulse = util.butterBandpassFilter(data=self.pulse,
                                               lowcut=fmin,
                                               highcut=fmax,
                                               fs=1/self.dt,
                                               order=order)
        self.spectrum = np.fft.fft(self.pulse)
        self.spectrum_plus = util.doFFT(np.flip(self.pulse))
        return self.pulse

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
        self.spectrum_plus = np.zeros(self.nSamples, dtype='complex')
        self.spectrum_plus[:self.nHalf] = self.spectrum[:self.nHalf]
        return self.pulse

    def do_impulse_response(self, IR, IR_freq): #TODO: APPLY SPECTRUM PLUS
        spectrum_shift = np.fft.fftshift(self.spectrum)
        freq_shift = np.fft.fftshift(self.freq_space)
        nFreq = len(freq_shift)
        IR_fmin = min(IR_freq)
        IR_fmax = max(IR_freq)
        nMid = util.findNearest(freq_shift, 0)

        freq_shift_positive = freq_shift[nMid:]
        freq_shift_negative = -1* np.flip(freq_shift[:nMid])
        nFreq_pos = len(freq_shift_positive)
        nFreq_neg = len(freq_shift_positive)
        self.IR = np.zeros(nFreq)
        self.IR_positive = np.zeros(nFreq_pos)
        self.IR_negative = np.zeros(nFreq_neg)
        for i in range(nFreq_pos):
            freq_plus = freq_shift_positive[i]
            j_neg = util.findNearest(freq_shift_negative, freq_plus)
            freq_neg = freq_shift_negative[j_neg]
            if freq_plus < IR_fmin:
                self.IR_positive[i] = 0
                self.IR_negative[j_neg] = 0
            elif freq_plus > IR_fmax:
                self.IR_positive[i] = 0
                self.IR_negative[j_neg] = 0
            elif freq_plus <= IR_fmax and freq_plus >= IR_fmin:
                print('reset')
                k_pos = util.findNearest(IR_freq, freq_plus)
                self.IR_positive[i] = IR[k_pos]
                self.IR_negative[j_neg] = IR[k_pos]
        self.IR[nMid:] = self.IR_positive
        self.IR[:nMid] = np.flip(self.IR_negative)

        spectrum_shift *= self.IR
        self.spectrum = np.fft.ifftshift(spectrum_shift)
        self.pulse = np.fft.ifft(self.spectrum)
        self.spectrum_plus = np.zeros(self.nSamples, dtype='complex')
        self.spectrum_plus[:self.nHalf] = self.spectrum[:self.nHalf]
        return self.pulse

    def set_psuedo_fmcw(self, fmin, fmax, IR, IR_freq):
        #Set Impulse
        jj = util.findNearest(self.tspace, self.t_centre)
        self.pulse = np.zeros(self.nSamples)
        self.pulse[jj] = self.amplitude
        self.spectrum = np.fft.fft(self.pulse)

        #Convert to FD
        spectrum_shift = np.fft.fftshift(self.spectrum)
        freq_space = np.fft.fftfreq(self.nSamples, self.dt)
        freq_space = np.fft.fftshift(freq_space)

        #Convert to POS AND NEG
        nMid = util.findNearest(freq_space, 0)
        freq_shift_positive = freq_space[nMid:]
        freq_shift_negative = -1* np.flip(freq_space[:nMid])
        spectrum_positive = spectrum_shift[nMid:]
        spectrum_negative = np.flip(spectrum_shift[:nMid])

        #CUT FD to Limits of Bandwidth
        ii_min_plus = util.findNearest(freq_shift_positive, fmin)
        ii_max_plus = util.findNearest(freq_shift_positive, fmax)
        nDelta_plus = ii_max_plus - ii_min_plus

        ii_min_minus = util.findNearest(freq_shift_negative, fmin)
        ii_max_minus = util.findNearest(freq_shift_negative, fmax)
        nDelta_minus = ii_max_minus - ii_min_minus

        spectrum_positive[:ii_min_plus] = 0
        spectrum_positive[ii_max_plus:] = 0
        #spectrum_positive[ii_min_plus:ii_max_plus] *= np.blackman(nDelta_plus) #
        spectrum_positive[ii_min_plus:ii_max_plus] *= tukey(nDelta_plus)
        spectrum_negative[:ii_min_minus] = 0
        spectrum_negative[ii_max_minus:] = 0
        #spectrum_negative[ii_min_minus:ii_max_minus] *= np.blackman(nDelta_minus) #
        spectrum_negative[ii_min_minus:ii_max_minus] *=tukey(nDelta_minus)
        nFreq_pos = len(freq_shift_positive)
        nFreq_neg = len(freq_shift_positive)
        nFreq = len(freq_space)
        IR_fmin = min(IR_freq)
        IR_fmax = max(IR_freq)
        self.IR = np.zeros(nFreq)
        self.IR_positive = np.zeros(nFreq_pos)
        self.IR_negative = np.zeros(nFreq_neg)
        for i in range(nFreq_pos):
            freq_plus = freq_shift_positive[i]
            j_neg = util.findNearest(freq_shift_negative, freq_plus)
            freq_neg = freq_shift_negative[j_neg]
            if freq_plus < IR_fmin:
                self.IR_positive[i] = 0
                self.IR_negative[j_neg] = 0
            elif freq_plus > IR_fmax:
                self.IR_positive[i] = 0
                self.IR_negative[j_neg] = 0
            elif freq_plus <= IR_fmax and freq_plus >= IR_fmin:
                print('reset')
                k_pos = util.findNearest(IR_freq, freq_plus)
                self.IR_positive[i] = IR[k_pos]
                self.IR_negative[j_neg] = IR[k_pos]
        self.IR[nMid:] = self.IR_positive
        self.IR[:nMid] = np.flip(self.IR_negative)

        spectrum_shift *= self.IR
        self.spectrum = np.fft.ifftshift(spectrum_shift)
        self.pulse = np.fft.ifft(self.spectrum)
        self.spectrum_plus = np.zeros(self.nSamples, dtype='complex')
        self.spectrum_plus[:self.nHalf] = self.spectrum[:self.nHalf]

        return self.pulse

    def hilbert_transform(self):
        tx_sig_r = self.pulse.real
        tx_sig_i = util.hilbertTransform(tx_sig_r)
        tx_sig_c = tx_sig_r + 1j*tx_sig_i
        self.pulse = tx_sig_c
        self.spectrum = np.fft.fft(tx_sig_r)
        self.spectrum_plus = np.zeros(self.nSamples, dtype='complex')
        self.spectrum_plus[:self.nHalf] = self.spectrum[:self.nHalf]

        return self.pulse

    def add_gaussian_noise(self):
        nSamples = len(self.spectrum)
        noise_amplitude = self.noise_amplitude*max(abs(self.spectrum))
        if noise_amplitude > 0:
            self.noise = np.random.normal(noise_amplitude, noise_amplitude/np.sqrt(noise_amplitude), nSamples)# * float(nSamples)
            #self.noise += util.hilbertTransform(np.random.normal(noise_amplitude, noise_amplitude, nSamples))# * float(nSamples)
            self.spectrum += self.noise
            self.pulse = np.fft.ifft(self.spectrum)
            self.spectrum_plus = util.doFFT(np.flip(self.pulse))

            #self.pulse += self.noise
            #self.spectrum = np.fft.fft(self.pulse)
    def add_white_noise(self):
        nSamples = len(self.spectrum)
        noise_amplitude = self.noise_amplitude*max(abs(self.spectrum))
        if noise_amplitude > 0:
            self.noise = noise_amplitude * np.random.uniform(0,1, nSamples)# * float(nSamples)
            self.spectrum += self.noise
            self.spectrum_plus = np.zeros(self.nSamples, dtype='complex')
            self.spectrum_plus[:self.nHalf] = self.spectrum[:self.nHalf]

            self.pulse = np.fft.ifft(self.spectrum)
            #self.pulse += self.noise
            #self.spectrum = np.fft.fft(self.pulse)
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