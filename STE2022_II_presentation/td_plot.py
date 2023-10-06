import datetime
import time

import numpy as np
import sys
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

sim0 = ppp.paraProp(iceLength=iceLength, iceDepth=iceDepth, dx=dx0, dz=dz, airHeight=airHeight)
sim1 = ppp.paraProp(iceLength=iceLength, iceDepth=iceDepth, dx=dx, dz=dz, airHeight=airHeight)

#fname = 'share/aletsch/aletsch_0604_max_peak_15m.txt'
fname = 'share/guliya.txt'

nprof_data = np.genfromtxt(fname)
z_profile = nprof_data[:,0]
n_profile = nprof_data[:,1]
sourceDepth = 6.5
rxDepth = 6.5

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
tx_signal0 = tx_signal(frequency=freq, bandwidth=band, t_centre=10, dt=dt, tmax=tmax)
tspace = tx_signal0.tspace
#tx_pulse = tx_signal0.get_gausspulse()
tx_pulse = tx_signal0.get_gausspulse_real()
sim1.set_td_source_signal(tx_pulse, dt=dt)

R_bh = 42
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

fig = pl.figure(figsize=(12,12),dpi=120)
ax1 = pl.subplot2grid((2,2), (0,0),colspan=2)
pmesh = ax1.imshow(20*np.log10(absu), aspect='auto', cmap='hot', vmin=-110, vmax=20, extent=(x[0], x[-1], z[-1], z[0]))
cbar = pl.colorbar(pmesh)
cbar.set_label('Power [dBu]')
ax1.set_title('Absolute Field, f = ' + str(freq*1e3) + ' MHz')
ax1.fill_betweenx(z, 10-0.04, 10+0.04, where=z>0,color='k')
ax1.fill_betweenx(z, 25-0.04, 25+0.04, where=z>0,color='k')
ax1.fill_betweenx(z, 42-0.04, 42+0.04, where=z>0,color='k')
ax1.scatter(0,sourceDepth,c='k')
ax1.scatter(R_bh, rxDepth, c='b')
ax1.set_xlabel('x [m]')
ax1.set_ylabel('z [m]')

ax2 = pl.subplot2grid((2,2),(1,0))
ax2.plot(n_profile, z_profile)
ax2.set_xlim(1,2)
ax2.axhline(sourceDepth,c='k')
ax2.axhline(rxDepth,c='b')

ax2.set_xlabel('n')
ax2.set_ylabel('z [m]')
ax2.set_ylim(15,-2)
ax2.grid()


ax3 = pl.subplot2grid((2,2), (1,1))
#ax3.plot(tspace, rx_pulse.real)
ax3_twin = ax3.twinx()
ax3_twin.plot(tspace, abs(tx_pulse)**2,c='k')
ax3.plot(tspace, abs(rx_pulse)**2,c='b')
ax3.grid()
ax3.set_xlabel('Time [ns]')
ax3.set_ylabel('Amplitude [u]')
pl.tight_layout()
#pl.savefig('td_pulse_with_tx_complex.png')

pl.show()