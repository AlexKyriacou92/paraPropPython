import os.path
import sys
import numpy as np
import time
import datetime
import h5py
import configparser
from numpy import exp, log
import matplotlib.pyplot as pl
from sys import argv
import util_nuRadioMC
sys.path.append('../')
from paraPropPython import paraProp as ppp
import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rxList_from_file, create_tx_signal, create_transmitter_array
from data import create_hdf_bscan, bscan_rxList, create_hdf_FT
import util
from data import ascan

nArgs = len(argv)
if nArgs == 2:
    fname_config = argv[1]
else:
    print('Input error, enter the config file for the analysis')
    print('For example: $ python', argv[0], 'config_analysis.txt')
    exit()

config = configparser.ConfigParser()
config.read(fname_config)

input_config = config['DATAFILES']
label_config = config['LABELS']
rx_config = config['RX']

fname_list = []
for key in input_config.keys():
    fname_k = input_config[key]
    fname_list.append(fname_k)
label_list = []
for key in label_config.keys():
    label_k = label_config[key]
    label_list.append(label_k)

x_rx = float(rx_config['X_RX'])
z_rx = float(rx_config['Z_RX'])
nSims = len(fname_list)
pulse_list = []
tspace_list = []
fspace_list = []
spec_list = []

colors = ['g', 'b', 'purple']
for i in range(nSims):
    ascan_i = ascan()
    fname_hdf_i = fname_list[i]

    ascan_i.load_from_hdf(fname_hdf_i)
    z_tx = ascan_i.tx_depths[0]
    print(i, fname_hdf_i)
    if i == 0:
        ascan_0 = ascan_i
        travel_times, amp_list, path_lengths, solution_types = util_nuRadioMC.get_ray_points(0, z_tx, x_rx, z_rx, 'analytic')

    pulse_rx = ascan_i.get_ascan(z_tx=z_tx, x_rx=x_rx, z_rx=z_rx)
    pulse_list.append(pulse_rx)
    tspace_list.append(ascan_i.tspace)

    spec_rx = ascan_i.get_spectrum(z_tx=z_tx, x_rx=x_rx, z_rx=z_rx)
    spec_rx = np.fft.fftshift(spec_rx)
    spec_list.append(spec_rx)
    fspace = np.fft.fftfreq(ascan_i.nSamples, ascan_i.dt)
    fspace = np.fft.fftshift(fspace)
    fspace_list.append(fspace)

if len(travel_times) >= 2:
    dt = travel_times[1]-travel_times[0]

fontsize = 18
labelsize = 14
tx_sig = ascan_0.tx_signal.pulse
fig = pl.figure(figsize=(8,12),dpi=100)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
#fig.suptitle('Source Signal',fontsize=fontsize)
ax1.set_title('Pulse Amplitude $V_{tx}$',fontsize=fontsize)
ax2.set_title('Spectrum Amplitude $A_{tx}$',fontsize=fontsize)

if len(tspace_list[0]) != len(tx_sig.real):
    tspace_tx = np.linspace(0, max(tspace_list[0]), len(tx_sig))
    ax1.plot(tspace_tx, tx_sig.real * 1000,c='k')
else:
    tspace_tx = tspace_list[0]
    ax1.plot(tspace_tx, tx_sig.real * 1000,c='k')
fspace_tx = np.fft.rfftfreq(len(tspace_tx), tspace_tx[1]-tspace_tx[0])
spec_tx = np.fft.rfft(tx_sig)

ax1.set_xlabel('Time t [ns]',fontsize=fontsize)
ax1.set_ylabel(r'$V_{tx}$ [mV/m]',fontsize=fontsize)
ax1.grid()
ax1.set_xlim(20,80)
ax1.tick_params(axis='both', labelsize=labelsize)

ax2.plot(fspace_tx, abs(spec_tx)*1000,c='k')
ax2.set_xlabel('Frequency f [GHz]',fontsize=fontsize)
ax2.set_ylabel(r'$A_{tx}$ [mV/m/GHz]',fontsize=fontsize)
ax2.grid()
ax2.tick_params(axis='both', labelsize=labelsize)
ax2.set_xlim(0,1)
fig.savefig('source_waveform.png', bbox_inches='tight')
pl.show()

fontsize=20
labelsize=16
t0 = 100.
fig = pl.figure(figsize=(10,6),dpi=150)
ax = fig.add_subplot(111)
title_str = 'Simulated Trace\n$z_{tx} =$ ' + str(z_tx) + ' m, '
title_str += '$x_{rx} = $ ' + str(x_rx) + ' m, $z_{rx} = $ ' + str(z_rx) + ' m'
ax.set_title(title_str, fontsize=fontsize)
for i in range(nSims):
    pulse_rx0 = pulse_list[i]
    ii_max = np.argmax(abs(pulse_rx0))
    tspace = tspace_list[i]
    ii0 = util.findNearest(tspace, t0)
    ii_delta = ii_max-ii0
    pulse_rx = np.roll(pulse_rx0, -ii_delta)
    if len(tspace) != len(pulse_rx):
        tspace_rx = np.linspace(0, max(tspace), len(pulse_rx))
    else:
        tspace_rx = tspace_list[i]

    ax.plot(tspace_rx, pulse_rx.real, label=label_list[i])

t_low = 0
t_high = (tspace[ii_max] + dt)*0.9

ax.tick_params(axis='both', labelsize=labelsize)
ax.set_xlabel('Time [ns]',fontsize=fontsize)
ax.set_ylabel('E field amplitude [mV]',fontsize=fontsize)
ax.legend(fontsize=fontsize)
ax.grid()
#ax.axvline(tspace[ii0],color='k')

if len(travel_times) >= 2:
    ax.axvline(tspace[ii0] + dt,color='k')

ax.set_xlim(0,1000)

prefix = 'rx_trace_ztx_' + str(int(z_tx)) + 'm_xrx_' + str(int(x_rx)) + 'm_zrx_' + str(int(z_rx)) + 'm'
fname_pulse = prefix + '_real.png'
fig.savefig(fname_pulse)

pl.show()

fig = pl.figure(figsize=(10,6),dpi=150)
ax = fig.add_subplot(111)
title_str = 'Simulated Trace\n$z_{tx} =$ ' + str(z_tx) + ' m, '
title_str += '$x_{rx} = $ ' + str(x_rx) + ' m, $z_{rx} = $ ' + str(z_rx) + ' m'
ax.set_title(title_str, fontsize=fontsize)
for i in range(nSims):
    pulse_rx0 = pulse_list[i]
    ii_max = np.argmax(abs(pulse_rx0))
    tspace = tspace_list[i]
    ii0 = util.findNearest(tspace, t0)
    ii_delta = ii_max-ii0
    pulse_rx = np.roll(pulse_rx0, -ii_delta)

    ax.plot(tspace, abs(pulse_rx), label=label_list[i])
ax.tick_params(axis='both', labelsize=labelsize)
ax.set_xlabel('Time [ns]',fontsize=fontsize)
ax.set_ylabel('E field amplitude [mV]',fontsize=fontsize)
ax.legend(fontsize=fontsize)
ax.grid()
ax.set_xlim(t_low, t_high)

fig.savefig('rx_pulse_abs.png')
pl.show()

fig = pl.figure(figsize=(10,6),dpi=150)
ax = fig.add_subplot(111)
title_str = 'Simulated Trace Spectrum \n$z_{tx} =$ ' + str(z_tx) + ' m, '
title_str += '$x_{rx} = $ ' + str(x_rx) + ' m, $z_{rx} = $ ' + str(z_rx) + ' m'
ax.set_title(title_str, fontsize=fontsize)
for i in range(nSims):
    ax.plot(fspace_list[i], abs(spec_list[i]),label=label_list[i])
ax.grid()
ax.tick_params(axis='both', labelsize=labelsize)
ax.set_xlabel('Frequency [GHz]',fontsize=fontsize)
ax.set_ylabel('E field amplitude [mV/Hz]',fontsize=fontsize)
ax.legend(fontsize=fontsize)
fig.savefig('rx_spectrum.png')
pl.show()