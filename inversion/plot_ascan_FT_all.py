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
from data import create_tx_signal, bscan, bscan_rxList, create_hdf_bscan, bscan_FT
from plotting_functions import compare_ascans, compare_ascans2, compare_ascans3, compare_ascan_FT_data
from objective_functions import misfit_function_ij, misfit_function_ij0

path2data = sys.argv[1]
path2sim = sys.argv[2]
path2plots = sys.argv[3]

parent_dir = os.path.dirname(path2sim)
print(parent_dir)

bscan_data = bscan_FT()
bscan_data.load_from_hdf(fname=path2data)
bscan_sim = bscan_rxList()
bscan_sim.load_sim(fname=path2sim)

tx_signal = bscan_sim.tx_signal
pl.plot(tx_signal.tspace, abs(tx_signal.pulse[:-1]))
ii_max = np.argmax(abs(tx_signal.pulse[:-1]))
print(tx_signal.tspace[ii_max])
pl.grid()
pl.show()

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

if os.path.isdir(path2plots) == False:
    os.system('mkdir ' + path2plots)

for i in range(len(bscan_data.txDepths)):
    z_tx = bscan_data.txDepths[i]
    x_rx = bscan_data.rxRanges[i]
    z_rx = bscan_data.rxDepths[i]
    ascan_data = bscan_data.get_ascan(z_tx, x_rx, z_rx)
    ascan_sim = bscan_sim.get_ascan_from_depth(z_tx, x_rx, z_rx)[:-1]
    #ascan_sim2 = bscan_sim.get_ascan_from_depth(z_tx, x_rx2, z_rx2)
    t_shift = 10

    sig_auto = abs(cc(ascan_data, ascan_data))

    ascan_shift = 1*np.roll(ascan_data, int(t_shift/bscan_data.dt))

    #path2plots = parent_dir + '/plots'
    compare_ascan_FT_data(bscan_data=bscan_data, bscan_sim=bscan_sim,z_tx=z_tx, x_rx=x_rx, z_rx=z_rx, mode_plot='abs', tmin=0, tmax=400, path2plot=path2plots)
    #compare_ascan_FT_data(bscan_data=bscan_data, bscan_sim=bscan_sim,z_tx=z_tx, x_rx=x_rx, z_rx=z_rx, mode_plot='envelope', tmin=0, tmax=600)
    '''
    m_ij = misfit_function_ij(ascan_data, ascan_sim,
                              tspace)
    print(m_ij, 1/m_ij)
    chi_local, chi_out, t_out, chi_sum = misfit_function_ij0(ascan_data, ascan_sim,tspace, mode=mode0,tmin=tmin0, tmax=tmax0)
    print('Simulation x Data')
    chi_time = misfit_function_ij(ascan_data, ascan_sim,tspace, mode='envelope')

    #cross_correl = cc(abs(ascan_data), abs(ascan_sim))
    cross_correl = cc(abs(ascan_data), abs(bscan_sim.tx_signal.pulse))
    cross_correl2 = cc(abs(ascan_sim), abs(bscan_sim.tx_signal.pulse))

    tspace_lag = np.linspace(-max(tspace), max(tspace), len(cross_correl))
    fig = pl.figure(figsize=(8,5),dpi=120)
    ax = fig.add_subplot(111)
    ax2 = ax.twiny()
    ax.plot(tspace_lag, cross_correl,label='FT',c='r')
    ax2.plot(tspace_lag, cross_correl2, label='Simulation',c='b')
    ax.legend('bottom right')
    ax2.legend()
    ax2.set_ylabel('Simulation A_tx * A_rx')
    ax.set_ylabel('FT Data A_tx * A_rx')
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('')
    ax.set_xlim(0, 600)
    ax2.set_xlim(0, 600)
    ax.grid()
    fname_plot = path2plots + '/ascan_z_tx=' + str(z_tx) + '_x_rx=' + str(x_rx) + '_z_rx=' + str(z_rx) + '_plot.png'
    fig.savefig(fname_plot)
    pl.close(fig)
    '''