import os.path
import sys
import numpy as np
import time
import datetime
import h5py
import configparser
from matplotlib import pyplot as pl
from makeDepthScan import depth_scan_from_hdf
from scipy.signal import correlate as cc

sys.path.append('../')

import util
import paraPropPython as ppp
from receiver import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rxList_from_file, create_transmitter_array, create_hdf_FT
from data import create_tx_signal, bscan, bscan_rxList, create_hdf_bscan
from plotting_functions import compare_ascans, compare_ascans2
from objective_functions import misfit_function_ij, misfit_function_ij0

path2data = sys.argv[1]
path2sim = sys.argv[2]
z_tx = float(sys.argv[3])
x_rx = float(sys.argv[4])
z_rx = float(sys.argv[5])

parent_dir = os.path.dirname(path2sim)
print(parent_dir)

bscan_data = bscan_rxList()
bscan_data.load_sim(fname=path2data)
bscan_sim = bscan_rxList()
bscan_sim.load_sim(fname=path2sim)

#sig_cc = cc(bscan_sim.get_ascan_from_depth(10, 25, 10), bscan_data.get_ascan_from_depth(10, 25, 10))
#tspace_cc = np.linspace(-max(bscan_sim.tspace), max(bscan_sim.tspace), len(sig_cc))
#pl.plot(tspace_cc, sig_cc-np.flip(sig_cc))
#pl.plot(tspace_cc, sig_cc)
#pl.plot(tspace_cc, np.flip(sig_cc))
#pl.show()

mode0 = 'Envelope'
tmin0 = 100
tmax0 = 300
z_rx2 = 15
x_rx2 = 42
tspace = bscan_sim.tspace
ascan_data = bscan_data.get_ascan_from_depth(z_tx, x_rx, z_rx)
ascan_sim = bscan_sim.get_ascan_from_depth(z_tx, x_rx, z_rx)
ascan_sim2 = bscan_sim.get_ascan_from_depth(z_tx, x_rx2, z_rx2)
t_shift = 5

sig_auto = abs(cc(ascan_data, ascan_data))

ascan_shift = 1*np.roll(ascan_data, int(t_shift/bscan_data.dt))

path2plots = parent_dir + '/plots/'
compare_ascans(bscan_data=bscan_data, bscan_sim=bscan_sim,z_tx=z_tx, x_rx=x_rx, z_rx=z_rx, mode_plot='pulse', tmin=80, tmax=280, path2plot=path2plots)

compare_ascans2(bscan_data=bscan_data, bscan_sim=bscan_sim,z_tx=z_tx, x_rx=x_rx, z_rx=z_rx, mode_plot='pulse', tmin=80, tmax=280, path2plot=path2plots)
m_ij = misfit_function_ij(ascan_data, ascan_sim,
                          tspace)
print(m_ij, 1/m_ij)
chi_local, chi_out, t_out, chi_sum = misfit_function_ij0(ascan_data, ascan_sim,tspace, mode=mode0,tmin=tmin0, tmax=tmax0)
print('Simulation x Data')
chi_time = misfit_function_ij(ascan_data, ascan_sim,tspace, mode='Correlation')


fig = pl.figure(figsize=(10,15), dpi=100)
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
ax2.plot(t_out, chi_out,c='b', label='Misfit = ' + str(round(chi_local,2)) + '\n' + 't_off = ' + str(chi_time))
sig_cc = abs(cc(ascan_data, ascan_sim))
tspace_cc = np.linspace(-max(tspace), max(tspace), len(sig_cc))
ax3.plot(tspace_cc, sig_cc/sig_auto, c='b')

print('Simulation2 x Data')

chi_local, chi_out, t_out, chi_sum = misfit_function_ij0(ascan_data, ascan_sim2,tspace, mode=mode0,tmin=tmin0, tmax=tmax0)
chi_time = misfit_function_ij(ascan_data, ascan_sim2,tspace, mode='Correlation')
ax2.plot(t_out, chi_out,c='g', label='Misfit = ' + str(round(chi_local,2)) + '\n' + 't_off = ' + str(chi_time))
sig_cc = abs(cc(ascan_data, ascan_sim2))
ax3.plot(tspace_cc, sig_cc/sig_auto, c='g')

print('Data x Data')

chi_local, chi_out, t_out, chi_sum = misfit_function_ij0(ascan_data, ascan_data,tspace, mode=mode0,tmin=tmin0, tmax=tmax0)
chi_time = misfit_function_ij(ascan_data, ascan_data,tspace, mode='Correlation')
sig_cc = abs(cc(ascan_data, ascan_data))
ax2.plot(t_out, chi_out,c='r', label='Misfit = ' + str(round(chi_local,2)) + '\n' + 't_off = ' + str(chi_time))
ax3.plot(tspace_cc, sig_cc/sig_auto,c='r')

ax1.plot(tspace, ascan_sim,c='b',label='Simulation',alpha=0.6)
ax1.plot(tspace, ascan_sim2,c='g',label='Simulation, x_rx = ' + str(x_rx2) + ' m, z_rx = ' + str(z_rx2) + ' m',alpha=0.6)
ax1.plot(bscan_data.tspace, ascan_data,c='r',label='Data')

print('Data x Data Shift')

chi_local, chi_out, t_out, chi_sum = misfit_function_ij0(ascan_data, ascan_shift,tspace, mode=mode0,tmin=tmin0, tmax=tmax0)
chi_time = misfit_function_ij(ascan_data,ascan_shift,tspace, mode='Correlation')

ax1.plot(tspace, ascan_shift,c='c',label='Data with delta_t = ' + str(t_shift),alpha=0.6)
ax2.plot(t_out, chi_out,c='c', label='Misfit = ' + str(round(chi_local,2)) + '\n' + 't_off = ' + str(chi_time))
sig_cc = abs(cc(ascan_data, ascan_shift))
ax3.plot(tspace_cc, sig_cc/sig_auto,c='c')
ax2.legend()
ax1.grid()
ax2.grid()
ax1.set_xlim(0,400)
ax2.set_xlim(0,400)
ax3.set_xlim(-400,400)
ax1.legend()
ax3.grid()
pl.show()