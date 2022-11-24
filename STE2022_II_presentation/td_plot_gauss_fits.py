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
dt = 1/f_sample

R_bh = 25
sourceDepth = 6.5
rxDepth = 6.5

#=========================================
hdf_data = h5py.File('Field-Test-data.h5')

rxRanges = np.array(hdf_data['rxRanges'])
rxDepths = np.array(hdf_data['rxDepths'])
fftArray = np.array(hdf_data['fftArray'])
jj_data = util.findNearest(rxDepths, sourceDepth)
tspace_data = np.array(hdf_data['tspace'])
print(jj_data)
#===========================================

sim0 = ppp.paraProp(iceLength=iceLength, iceDepth=iceDepth, dx=dx0, dz=dz, airHeight=airHeight)
sim1 = ppp.paraProp(iceLength=iceLength, iceDepth=iceDepth, dx=dx, dz=dz, airHeight=airHeight)

fname = 'share/aletsch/aletsch_0604_max_peak_15m.txt'
#fname = 'share/guliya.txt'

nprof_data = np.genfromtxt(fname)
z_profile = nprof_data[:,0]
n_profile = nprof_data[:,1]


sim0.set_n(zVec=z_profile, nVec=n_profile)
sim0.set_dipole_source_profile(freq, sourceDepth)
sim1.set_n(zVec=z_profile, nVec=n_profile)
sim1.set_dipole_source_profile(freq, sourceDepth)

sim0.set_cw_source_signal(freq)

tstart = time.time()
sim0.do_solver()
tend = time.time()
fd_duration = tend - tstart

x = sim0.get_x()
z = sim0.get_z()

absu = abs(np.transpose(sim0.get_field()))

#PULSE
tmax = iceLength * max(n_profile) / util.c_light
t_centre = 10
tx_signal0 = tx_signal(frequency=freq, bandwidth=band, t_centre=t_centre, dt=dt, tmax=tmax)
tspace = tx_signal0.tspace
#tx_pulse = tx_signal0.get_gausspulse()
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
estimated_duration = nCalc * fd_duration * (dx0/dx)
print('Estimated pulse simulation time:', datetime.timedelta(seconds=estimated_duration))

sim1.do_solver(rxList=rxList, freqMin=freqMin, freqMax=freqMax)
rx_out = rxList[0]
rx_spectrum = rx_out.spectrum
rx_spectrum_w = np.zeros(tx_signal0.nSamples, dtype='complex')
weights = np.blackman(nCalc)
rx_spectrum_w[ii_min:ii_max] = weights * rx_spectrum[ii_min:ii_max]
rx_pulse = np.flip(util.doIFFT(rx_spectrum_w))
#rx_pulse = np.flip(util.doIFFT(rx_spectrum))
#rx_pulse = rx_out.get_signal()

k_shift = int(t_centre/dt)
rx_pulse = np.roll(rx_pulse, -k_shift)
rx_pulse_mag = abs(rx_pulse)
rx_pulse_power = rx_pulse_mag**2

indices = pku.indexes(rx_pulse_power, thres=0.1)
nPeaks = len(indices)



fig = pl.figure(figsize=(12,12),dpi=120)
ax = fig.add_subplot(111)


ax.plot(tspace, rx_pulse_power)

t_res = 1/band
rx_pulse_power2 = rx_pulse_power
nSamples = len(rx_pulse_power)
print(t_res)
rx_pulse_multi_fit = np.zeros(nSamples)

t_res_fit = t_res
for i in range(nPeaks):
    ii = indices[i]
    t_peak = tspace[ii]
    p_peak = rx_pulse_power2[ii]

    jj_cut1 = util.findNearest(tspace, t_peak - t_res)
    jj_cut2 = util.findNearest(tspace, t_peak + t_res)

    p0 = [t_peak, t_res, p_peak]
    p1 = scipy.optimize.least_squares(errfunc_simple, p0[:], args=(tspace[jj_cut1:jj_cut2], rx_pulse_power2[jj_cut1:jj_cut2]))

    gauss_fit = gaussian_simple(tspace, p1.x[0], p1.x[1], p1.x[2])
    rx_pulse_power2 -= gauss_fit
    #ax.plot(tspace, gauss_fit, c='k')
    print('peak_fit', p1.x[0], 'resolution:',p1.x[1], 'ns')
    rx_pulse_multi_fit += gauss_fit
    t_res_fit = p1.x[1]
ax.plot(tspace, rx_pulse_multi_fit,c='k')

rx_FFT_data = fftArray[jj_data]
rx_FFT_power = abs(rx_FFT_data)**2

pks_fft = pku.indexes(rx_FFT_power, thres=0.1)
nPeaks_fft = len(pks_fft)
print(nPeaks_fft)

ax2 = ax.twinx()
ax2.plot(tspace_data*1e9, rx_FFT_power,c='r')
rx_FFT_multi_fit = np.zeros(len(rx_FFT_power))
rx_FFT_power2 = rx_FFT_power

for i in range(nPeaks):
    ii = pks_fft[i]
    t_peak = tspace_data[ii]
    p_peak = rx_FFT_power2[ii]
    t_res2 = t_res_fit*1e-9


    jj_cut1 = util.findNearest(tspace_data, t_peak - t_res2)
    jj_cut2 = util.findNearest(tspace_data, t_peak + t_res2)

    p0 = [t_peak, t_res2, p_peak]
    print(p0)
    p1 = scipy.optimize.least_squares(errfunc_simple, p0[:], args=(tspace_data[jj_cut1:jj_cut2], rx_FFT_power2[jj_cut1:jj_cut2]))

    gauss_fit = gaussian_simple(tspace_data, p1.x[0], p1.x[1], p1.x[2])
    rx_FFT_power2 -= gauss_fit
    #ax.plot(tspace, gauss_fit, c='k')
    print('peak_fit', p1.x[0]*1e9, 'resolution:',p1.x[1]*1e9, 'ns')
    rx_FFT_multi_fit += gauss_fit
ax2.plot(tspace_data*1e9, rx_FFT_multi_fit,c='g')

ax.grid()
pl.savefig('compare-fft-rx-pulse.png')
pl.close()

def get_max_peak(tspace, rx_power):
    ii_max = np.argmax(rx_power)
    return tspace[ii_max]

#print(len(rx_FFT_power), len(rx_pulse_power))
nSamples_data = len(rx_FFT_power)

rx_FFT_power /= sum(rx_FFT_power)
rx_pulse_power /= sum(rx_pulse_power)

f_interp = interp1d(tspace, rx_pulse_power)
rx_pulse_interp = np.zeros(nSamples_data)

rx_pulse_interp[0] = rx_FFT_power[0]
rx_pulse_interp[-1] = rx_FFT_power[-1]
rx_pulse_interp[1:-1] = f_interp(tspace_data[1:-1])

S_list = []
t_list = []
Chi_list = []
t_pk0 = get_max_peak(tspace_data, rx_pulse_interp)
rx_pulse_interp2 = np.roll(rx_pulse_interp, int(t_pk0/(tspace_data[1]-tspace_data[0])))
for j in range(nSamples_data):
    rx_pulse_shift = np.roll(rx_pulse_interp2, j)

    t_pk_sim = get_max_peak(tspace_data, rx_pulse_shift)
    t_pk_data = get_max_peak(tspace_data, rx_FFT_power)

    Chi_sq = sum((rx_pulse_shift-rx_FFT_power)**2)
    Chi_list.append(Chi_sq)
    S_fitness = 1/Chi_sq
    delta_t = t_pk_sim- t_pk_data
    print(delta_t, S_fitness, Chi_sq)

    S_list.append(S_fitness)
    t_list.append(delta_t)

t_arr = np.array(t_list)
S_arr = np.array(S_list)
Chi_arr = np.array(Chi_list)

fig = pl.figure(figsize=(8,5),dpi=100)
ax1 = fig.add_subplot(111)
ax1.plot(t_arr, S_arr - np.mean(S_arr))
ax1.grid()
pl.savefig('S_fitness.png')
pl.show()

fig = pl.figure(figsize=(8,5),dpi=100)
ax1 = fig.add_subplot(111)
ax1.plot(t_arr, Chi_arr)
ax1.grid()
pl.savefig('Chi_fitness.png')
pl.close()