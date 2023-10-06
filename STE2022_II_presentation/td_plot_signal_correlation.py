import datetime
import time

import numpy as np
import sys

import scipy.optimize
from scipy.interpolate import interp1d

sys.path.append('../')

import paraPropPython as ppp
from receiver import receiver
from data import bscan
from transmitter import tx_signal

import scipy.signal as signal
import util
import h5py
import matplotlib.pyplot as pl
import peakutils as pku
import sys
from util import gaussian, fitfunc, errfunc, gaussian_simple, errfunc_simple, fitfunc_simple

R_bh = 25
sourceDepth = 6.5
rxDepth = 6.5

#=========================================
hdf_data = h5py.File('Field-Test-data.h5')

rxRanges = np.array(hdf_data['rxRanges'])
rxDepths = np.array(hdf_data['rxDepths'])
fftArray = np.array(hdf_data['fftArray'])
jj_data = util.findNearest(rxDepths, sourceDepth)
tspace_data = np.array(hdf_data['tspace']) * 1e9

rx_FFT_data = fftArray[jj_data]
rx_FFT_mag = abs(rx_FFT_data)
rx_FFT_power = abs(rx_FFT_data)**2
rx_FFT_norm = rx_FFT_data #/sum(rx_FFT_mag)

print(jj_data)

hdf_data.close()

dt_data = abs(tspace_data[1]-tspace_data[0])
f_sample = 1/dt_data
dt = 1/(f_sample)

#===========================================

iceLength = 50
iceDepth = 15.0
dx0 = 0.2
dx = 1
dz = 0.02
airHeight = 5.0

freq_1 = 1.3
freq_2 = 1.8
freq = freq_2

band = 0.2
f_nyq = 2.0
f_sample = 2*f_nyq

sim1 = ppp.paraProp(iceLength=iceLength, iceDepth=iceDepth, dx=dx, dz=dz, airHeight=airHeight)

fname = 'share/aletsch/aletsch_0604_max_peak_15m.txt'

nprof_data = np.genfromtxt(fname)
z_profile = nprof_data[:,0]
n_profile = nprof_data[:,1]


sim1.set_n(zVec=z_profile, nVec=n_profile)
sim1.set_dipole_source_profile(freq, sourceDepth)

x = sim1.get_x()
z = sim1.get_z()

#PULSE
#tmax = iceLength * max(n_profile) / util.c_light
tmax = 600
t_centre = 10
tx_signal0 = tx_signal(frequency=freq, bandwidth=band, t_centre=t_centre, dt=dt, tmax=tmax)
tspace = tx_signal0.tspace
tx_pulse = tx_signal0.get_gausspulse_real()
sim1.set_td_source_signal(tx_pulse, dt=dt)

rxList = []
rx1 = receiver(x=R_bh, z=rxDepth)
rxList.append(rx1)

freqMin = freq-band
freqMax = freq+band
ii_max = util.findNearest(sim1.freq, freqMax)
ii_min = util.findNearest(sim1.freq, freqMin)
nCalc = ii_max - ii_min

sim1.do_solver(rxList=rxList, freqMin=freqMin, freqMax=freqMax)
rx_out = rxList[0]
rx_spectrum = rx_out.spectrum
rx_spectrum_w = np.zeros(tx_signal0.nSamples, dtype='complex')
weights = np.blackman(nCalc)
rx_spectrum_w[ii_min:ii_max] = weights * rx_spectrum[ii_min:ii_max]
rx_pulse = np.flip(util.doIFFT(rx_spectrum_w))
rx_pulse_mag = abs(rx_pulse)
rx_pulse_power = rx_pulse_mag**2
rx_pulse_norm = rx_pulse #/sum(rx_pulse_mag)

def signal_correlation(sig1,sig2):
    sig_multi = abs(sig1 * sig2)
    sig_multi_sq = sig_multi**2
    S = sum(sig_multi_sq)
    return S

def get_max_peak(tspace, rx_power):
    ii_max = np.argmax(rx_power)
    return tspace[ii_max]

t_pk_data = get_max_peak(tspace_data, rx_FFT_power)
t_pk_sim = get_max_peak(tspace, rx_pulse_power)
t_offset = t_pk_data-t_pk_sim
print('Data maximum', t_pk_data, 'Sim maximum', t_pk_sim)
print(t_offset)

nSamples_sim = len(rx_pulse_norm)
nSamples_data = len(rx_FFT_norm)
dN = nSamples_data - nSamples_sim
if dN > 0:
    rx_FFT_norm = rx_FFT_norm[:-dN]
    tspace_data = tspace_data[:-dN]
elif dN < 0:
    rx_pulse_norm = rx_pulse_norm[:-dN]
    tspace = tspace[:-dN]
S_data_sim = signal_correlation(rx_FFT_norm, rx_pulse_norm)
print(t_offset, S_data_sim)

pl.figure(figsize=(12,8),dpi=120)
pl.plot(tspace_data, abs(rx_FFT_norm))
pl.plot(tspace, abs(rx_pulse_norm))
pl.grid()
pl.savefig('signal_correlation.png')
pl.show()


ii_offset = int(t_offset/dt_data)
rx_pulse_shift = np.roll(rx_pulse_norm, ii_offset)

pl.figure(figsize=(12,8),dpi=120)
pl.plot(tspace_data, abs(rx_FFT_norm))
pl.plot(tspace, abs(rx_pulse_shift))
pl.grid()
pl.savefig('signal_correlation_aligned.png')

pl.show()

print(signal_correlation(rx_FFT_norm, rx_pulse_shift))

ii_offset = -2*int(t_offset/dt_data)
rx_pulse_shift = np.roll(rx_pulse_norm, ii_offset)

pl.figure(figsize=(12,8),dpi=120)
pl.plot(tspace_data, abs(rx_FFT_norm))
pl.plot(tspace, abs(rx_pulse_shift))
pl.savefig('signal_correlation_unaligned.png')

pl.grid()

pl.show()

print(signal_correlation(rx_FFT_norm, rx_pulse_shift))

nPoints = 100
tspace2 = np.linspace(-100, 400, nPoints)
S_arr = np.zeros(nPoints)
for j in range(nPoints):
    dt_move_sim = tspace2[j] + t_offset
    ii_roll = int(dt_move_sim/dt_data)
    rx_pulse_shift = np.roll(rx_pulse_norm, ii_roll)
    S_arr[j] = signal_correlation(rx_FFT_norm, rx_pulse_shift)
pl.figure(figsize=(12,8),dpi=120)
pl.plot(tspace2, S_arr)
pl.ylabel(r'$S = \sum |A_{sim} - A_{data}|^{2}$')
pl.xlabel(r'Peak Offset $\Delta t$ [ns[]')
pl.grid()
pl.yscale('log')
pl.savefig('Signal_Correlation.png')
pl.show()
