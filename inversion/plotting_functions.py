import os.path
import sys
import numpy as np
import time
import datetime
import h5py
import configparser
from matplotlib import pyplot as pl
from scipy.signal import correlate

from makeDepthScan import depth_scan_from_hdf
from objective_functions import misfit_function_ij

sys.path.append('../')

import util
import paraPropPython as ppp
from receiver import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rxList_from_file, create_transmitter_array, create_hdf_FT
from data import create_tx_signal, bscan, bscan_rxList, create_hdf_bscan

def plot_ascan(bscan_sim, z_tx, x_rx, z_rx, tmin=None, tmax=None, mode_plot = 'pulse', path2plot=None):
    ascan_sim = bscan_sim.get_ascan_from_depth(z_tx=z_tx, x_rx=x_rx, z_rx=z_rx)
    tspace = bscan_sim.tspace
    tx_sig = bscan_sim.tx_signal
    fig = pl.figure(figsize=(8, 5), dpi=120)

    fig_label = '$f_{central}$ = ' + str(tx_sig.frequency*1e3) + ' MHz B = ' + str(tx_sig.bandwidth*1e3) + ' MHz \n'
    fig_label += '$z_{tx} = $ ' + str(z_tx) + ' m, $R = $ ' + str(x_rx) + ' m $z_{rx} = $ ' + str(z_rx) + ' m'

    ax = fig.add_subplot(111)
    ax.set_title(fig_label)
    if mode_plot == 'pulse':
        ax.plot(tspace, ascan_sim.real, label='Simulation', c='b', alpha=0.8)
    elif mode_plot == 'abs':
        ax.plot(tspace, abs(ascan_sim), label='Simulation', c='b', alpha=0.8)
    elif mode_plot == 'correlation':
        pulse_tx = tx_sig.pulse
        sig_sim_correl = correlate(ascan_sim, pulse_tx)
        tspace_lag = np.linspace(-max(tspace), max(tspace), len(sig_sim_correl))
        j_cut = util.findNearest(tspace_lag, 0)
        tspace_cut = tspace_lag[j_cut:]
        sig_sim_correl = abs(sig_sim_correl[j_cut:])
        ax.plot(tspace_cut, sig_sim_correl, c='b', label='Simulation', alpha=0.8)


    ax.grid()
    ax.legend()
    ax.set_ylabel('Amplitude A [u]')
    ax.set_xlabel('Time t [ns]')
    if tmin != None and tmax != None:
        ax.set_xlim(tmin, tmax)
    elif tmin == None and tmax != None:
        ax.set_xlim(min(tspace), tmax)
    elif tmin != None and tmax == None:
        ax.set_xlim(tmin, max(tspace))
    if path2plot != None:
        if os.path.isdir(path2plot) == False:
            os.system('mkdir ' + path2plot)
        fname_ascan = path2plot + '/'
        fname_ascan += 'ascan_z_tx=' + str(z_tx) + '_x_rx=' + str(x_rx) + '_z_rx=' + str(z_rx) + '_plot.png'
        fig.savefig(fname_ascan)
    pl.show()

def compare_ascans(bscan_data, bscan_sim, z_tx, x_rx, z_rx, tmin=None, tmax=None, mode_plot = 'pulse', path2plot=None):
    ascan_sim = bscan_sim.get_ascan_from_depth(z_tx=z_tx, x_rx=x_rx, z_rx=z_rx)
    ascan_data = bscan_data.get_ascan_from_depth(z_tx=z_tx, x_rx=x_rx, z_rx=z_rx)
    tx_sig = bscan_sim.tx_signal
    tspace = bscan_data.tspace

    fig = pl.figure(figsize=(8,5),dpi=120)
    ax = fig.add_subplot(111)


    fig_label = '$f_{central}$ = ' + str(tx_sig.frequency*1e3) + ' MHz B = ' + str(tx_sig.bandwidth*1e3) + ' MHz \n'
    fig_label += '$z_{tx} = $ ' + str(z_tx) + ' m, $R = $ ' + str(x_rx) + ' m $z_{rx} = $ ' + str(z_rx) + ' m'
    m_ij = misfit_function_ij(ascan_data, ascan_sim, tspace)
    m_ij0 = misfit_function_ij(ascan_data, np.roll(ascan_data, 3), tspace)
    # print(m_ij0)
    fig_label += '\n$m_{ij} =$ ' + str(round(m_ij, 2)) + '$s_{ij} = $' + str(round(1 / m_ij, 5))
    ax.set_title(fig_label)

    if mode_plot == 'pulse':
        ax.plot(tspace, ascan_data.real, label='PseudoData',c='r')
        ax.plot(tspace, ascan_sim.real, label='Simulation',c='b',alpha=0.8)
    elif mode_plot == 'abs':
        ax.plot(tspace, abs(ascan_data), label='PseudoData',c='r')
        ax.plot(tspace, abs(ascan_sim), label='Simulation',c='b',alpha=0.8)
    elif mode_plot == 'correlation':
        sig_tx = bscan_sim.tx_signal.pulse
        sig_sim_correl = correlate(ascan_sim, sig_tx)
        sig_data_correl = correlate(ascan_data, sig_tx)
        tspace_lag = np.linspace(-max(tspace), max(tspace), len(sig_sim_correl))
        j_cut = util.findNearest(tspace_lag, 0)
        tspace_cut = tspace_lag[j_cut:]
        sig_sim_correl = abs(sig_sim_correl[j_cut:])
        sig_data_correl = abs(sig_data_correl[j_cut:])
        ax.plot(tspace_cut, sig_data_correl, c='r', label='PseudoData')
        ax.plot(tspace_cut, sig_sim_correl, c='b', label='Simulation',alpha=0.8)
    elif mode_plot == 'envelope':
        E_data = np.sqrt(ascan_data.real**2 + ascan_data.imag**2)
        E_sim = np.sqrt(ascan_sim.real**2 + ascan_sim.imag**2)
        ax.plot(tspace, E_data, label='PseudoData', c='r')
        ax.plot(tspace, E_sim, label='Simulation', c='b', alpha=0.8)

    ax.grid()
    ax.legend()
    ax.set_ylabel('Amplitude A [u]')
    ax.set_xlabel('Time t [ns]')
    if tmin != None and tmax != None:
        ax.set_xlim(tmin, tmax)
    elif tmin == None and tmax != None:
        ax.set_xlim(min(tspace), tmax)
    elif tmin != None and tmax == None:
        ax.set_xlim(tmin, max(tspace))
    if path2plot != None:
        if os.path.isdir(path2plot) == False:
            os.system('mkdir ' + path2plot)
        fname_ascan = path2plot + '/'
        fname_ascan += 'ascan_compare_z_tx=' + str(z_tx) + '_x_rx=' + str(x_rx) + '_z_rx=' + str(z_rx) + '_plot.png'
        fig.savefig(fname_ascan)
    pl.show()