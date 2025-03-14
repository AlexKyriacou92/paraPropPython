import datetime
import numpy as np
import sys
import time
from os.path import join

import pylab as pl


sys.path.append('../')
import paraPropPython as ppp
from receiver import receiver
from data import ascan
import util
from transmitter import tx_signal

path2data = 'ascan_dir'
fname_data = 'example_ascan_ztx_50.0_Xice_100.0_Zice_100.0.h5'
fname_data = join(path2data, fname_data)
ascan_example = ascan()
ascan_example.load_from_hdf(fname_hdf=fname_data)

#Transmitter (TX) signal
tx_signal_1 = ascan_example.tx_signal # Pulse at Transmitter (complex)
tx_pulse = tx_signal_1.pulse
spec_tx = tx_signal_1.spectrum
fspace_tx = tx_signal_1.freq_space

tspace = ascan_example.tspace # Vector of times (waveform)
nSamples = ascan_example.nSamples # Number of samples in waveform
dt = ascan_example.dt #Time interval of sample

#TX depths
tx_depths = ascan_example.tx_depths
nTx = len(tx_depths)
t_centre = tx_signal_1.t_centre

# List of Receivers
rxList = ascan_example.rxList

n_profile = ascan_example.n_profile # Ref Index Profile
z_profile = ascan_example.z_profile # Depth vector - correpsonds to ref index profile

rxPulses = ascan_example.ascan_array # Array of Pulses: nTx x nRx x nSamples

# Example Plots
fontsize = 14
labelsize = 12

# Plots the Ref Index Profile and TX depths
fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
ax.set_title('Ref Index Profile $n(z)$ and TX depths $z_{tx}$',fontsize=fontsize)
ax.plot(z_profile, n_profile,c='k')
for i in range(nTx):
    if i == 0:
        ax.axvline(tx_depths[i],color='b',linestyle='--',label='TX')
ax.legend(fontsize=labelsize)
ax.grid()
ax.set_xlabel('Depths $z$ [m]',fontsize=fontsize)
ax.set_ylabel('Ref. Index $n(z)$',fontsize=fontsize)
ax.tick_params(axis='both',labelsize=labelsize)
pl.show()

#Pick a receiver you'd like to plot, i.e. at x = 90 m, z = 50 m
x_example = 90.
z_example = 50.

# Find the nearest receiver to your coordinates
ii_rx = util.get_rx_id(x_example, z_example, rxList) # finds index of nearest receiver from the list rxList
rx_true = rxList[ii_rx] # receiver object (nearest receiver)
x_true = rx_true.x # nearest receiver's range
z_true = rx_true.z # nearest receiver's depth

# Plots the signal transmitted from TX

fig = pl.figure(figsize=(12,6),dpi=120)
fig.suptitle('Pulse transmitted from TX',fontsize=fontsize)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.set_title('Pulse trace $A_{tx}(t)$',fontsize=fontsize)
ax1.plot(tspace, tx_pulse, c='b')
ax1.set_xlabel('Time t [ns]',fontsize=fontsize)
ax1.set_ylabel('$A_{tx}$ [V/m]',fontsize=fontsize)
ax1.grid()
ax1.tick_params(axis='both',labelsize=labelsize)
ax1.set_xlim(0, 5*t_centre)

ax2.set_title('Spectrum  $S_{tx}(f)$',fontsize=fontsize)
ax2.plot(fspace_tx, abs(spec_tx), c='b')
ax2.set_xlabel('Frequency f [GHz]',fontsize=fontsize)
ax2.set_ylabel('$S_{tx}$ [V/m]',fontsize=fontsize)
ax2.grid()
ax2.set_xlim(0, max(fspace_tx))
ax2.tick_params(axis='both', labelsize=labelsize)
pl.show()

pulse_rx = ascan_example.get_ascan(z_tx=tx_depths[0], x_rx=x_true, z_rx=z_true)
# Plots Pulse and Spectrum of the signal at the selected receiver
fig = pl.figure(figsize=(12,6),dpi=120)
fig.suptitle('Signal at RX(' + str(x_true) + 'm, ' + str(z_true) + 'm)')

rx_spec = ascan_example.get_spectrum(z_tx=tx_depths[0], x_rx=x_true, z_rx=z_true)
fspace = np.fft.fftfreq(nSamples, dt)

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.set_title('Pulse trace $A_{rx}(t)$',fontsize=fontsize)
ax1.plot(tspace, pulse_rx.real,c='m')
ax1.set_xlabel('Time t [ns]',fontsize=fontsize)
ax1.set_ylabel('$A_{rx}$ [V/m]',fontsize=fontsize)
ax1.tick_params(axis='both', labelsize=labelsize)
ax1.grid()

ax2.set_title('Spectrum  $S_{rx}(f)$',fontsize=fontsize)
ax2.plot(fspace, abs(rx_spec),c='m')
ax2.set_xlabel('Frequency f [GHz]',fontsize=fontsize)
ax2.set_ylabel('$S_{rx}$ [V/m/Hz]',fontsize=fontsize)
ax2.set_xlim(0, max(fspace))
ax2.grid()
ax2.tick_params(axis='both', labelsize=labelsize)
pl.show()