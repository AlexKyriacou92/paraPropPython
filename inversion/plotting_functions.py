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
        ax.plot(tspace, ascan_sim.real, label=label_sim, c='b', alpha=0.8)
    elif mode_plot == 'abs':
        ax.plot(tspace, abs(ascan_sim), label=label_sim, c='b', alpha=0.8)
    elif mode_plot == 'correlation':
        pulse_tx = tx_sig.pulse
        sig_sim_correl = correlate(ascan_sim, pulse_tx)
        tspace_lag = np.linspace(-max(tspace), max(tspace), len(sig_sim_correl))
        j_cut = util.findNearest(tspace_lag, 0)
        tspace_cut = tspace_lag[j_cut:]
        sig_sim_correl = abs(sig_sim_correl[j_cut:])
        ax.plot(tspace_cut, sig_sim_correl, c='b', label=label_sim, alpha=0.8)


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
    fig_label += '\n$m_{ij} =$ ' + str(round(m_ij, 2)) + ' $s_{ij} = $' + str(round(1 / m_ij, 5))
    ax.set_title(fig_label)

    parent_dir1 = os.path.dirname(bscan_sim.fname)
    parent_dir2 = os.path.dirname(bscan_data.fname)
    N_str1 = len(parent_dir1)
    N_str2 = len(parent_dir2)
    label_sim = 'Simulation ' + bscan_sim.fname[N_str1 + 1:-3]
    label_data = 'PseudoData ' + bscan_data.fname[N_str2 + 1:-3]

    if mode_plot == 'pulse':
        ax.plot(tspace, ascan_data.real, label=label_data,c='r')
        ax.plot(tspace, ascan_sim.real, label=label_sim,c='b',alpha=0.8)
    elif mode_plot == 'abs':
        ax.plot(tspace, abs(ascan_data), label=label_data,c='r')
        ax.plot(tspace, abs(ascan_sim), label=label_sim,c='b',alpha=0.8)
    elif mode_plot == 'correlation':
        sig_tx = bscan_sim.tx_signal.pulse
        sig_sim_correl = correlate(ascan_sim, sig_tx)
        sig_data_correl = correlate(ascan_data, sig_tx)
        tspace_lag = np.linspace(-max(tspace), max(tspace), len(sig_sim_correl))
        j_cut = util.findNearest(tspace_lag, 0)
        tspace_cut = tspace_lag[j_cut:]
        sig_sim_correl = abs(sig_sim_correl[j_cut:])
        sig_data_correl = abs(sig_data_correl[j_cut:])
        ax.plot(tspace_cut, sig_data_correl, c='r', label=label_data)
        ax.plot(tspace_cut, sig_sim_correl, c='b', label=label_sim,alpha=0.8)
    elif mode_plot == 'envelope':
        E_data = np.sqrt(ascan_data.real**2 + ascan_data.imag**2)
        E_sim = np.sqrt(ascan_sim.real**2 + ascan_sim.imag**2)
        ax.plot(tspace, E_data, label=label_data, c='r')
        ax.plot(tspace, E_sim, label=label_sim, c='b', alpha=0.8)

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
def compare_ascans2(bscan_data, bscan_sim, z_tx, x_rx, z_rx, tmin=None, tmax=None, mode_plot = 'pulse', path2plot=None):
    ascan_sim = bscan_sim.get_ascan_from_depth(z_tx=z_tx, x_rx=x_rx, z_rx=z_rx)
    ascan_data = bscan_data.get_ascan_from_depth(z_tx=z_tx, x_rx=x_rx, z_rx=z_rx)
    tx_sig = bscan_sim.tx_signal
    tspace = bscan_data.tspace

    fig = pl.figure(figsize=(20,8),dpi=120)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    fig_label = '$f_{central}$ = ' + str(tx_sig.frequency*1e3) + ' MHz B = ' + str(tx_sig.bandwidth*1e3) + ' MHz \n'
    fig_label += '$z_{tx} = $ ' + str(z_tx) + ' m, $R = $ ' + str(x_rx) + ' m $z_{rx} = $ ' + str(z_rx) + ' m'
    m_ij = misfit_function_ij(ascan_data, ascan_sim, tspace,mode='Waveform')
    # m_ij0 = misfit_function_ij(ascan_data, np.roll(ascan_data, 3), tspace)
    # print(m_ij0)
    fig_label += '\n$m_{ij} =$ ' + str(round(m_ij, 2)) + '$s_{ij} = $' + str(round(1 / m_ij, 5))
    ax1.set_title(fig_label)

    parent_dir1 = os.path.dirname(bscan_sim.fname)
    parent_dir2 = os.path.dirname(bscan_data.fname)
    N_str1 = len(parent_dir1)
    N_str2 = len(parent_dir2)
    label_sim = 'Simulation ' + bscan_sim.fname[N_str1 + 1:]
    label_data = 'PseudoData ' + bscan_data.fname[N_str2 + 1:]

    if mode_plot == 'pulse':
        ax1.plot(tspace, ascan_data.real, label=label_data,c='r')
        ax1.plot(tspace, ascan_sim.real, label=label_sim,c='b',alpha=0.8)
    elif mode_plot == 'abs':
        ax1.plot(tspace, abs(ascan_data), label=label_data,c='r')
        ax1.plot(tspace, abs(ascan_sim), label=label_sim,c='b',alpha=0.8)
    elif mode_plot == 'correlation':
        sig_tx = bscan_sim.tx_signal.pulse
        sig_sim_correl = correlate(ascan_sim, sig_tx)
        sig_data_correl = correlate(ascan_data, sig_tx)
        tspace_lag = np.linspace(-max(tspace), max(tspace), len(sig_sim_correl))
        j_cut = util.findNearest(tspace_lag, 0)
        tspace_cut = tspace_lag[j_cut:]
        sig_sim_correl = abs(sig_sim_correl[j_cut:])
        sig_data_correl = abs(sig_data_correl[j_cut:])
        ax1.plot(tspace_cut, sig_data_correl, c='r', label=label_data)
        ax1.plot(tspace_cut, sig_sim_correl, c='b', label=label_sim,alpha=0.8)
    elif mode_plot == 'envelope':
        E_data = np.sqrt(ascan_data.real**2 + ascan_data.imag**2)
        E_sim = np.sqrt(ascan_sim.real**2 + ascan_sim.imag**2)
        ax1.plot(tspace, E_data, label=label_data, c='r')
        ax1.plot(tspace, E_sim, label=label_sim, c='b', alpha=0.8)

    ax1.grid()
    ax1.legend()
    ax1.set_ylabel('Amplitude A [u]')
    ax1.set_xlabel('Time t [ns]')
    if tmin != None and tmax != None:
        ax1.set_xlim(tmin, tmax)
    elif tmin == None and tmax != None:
        ax1.set_xlim(min(tspace), tmax)
    elif tmin != None and tmax == None:
        ax1.set_xlim(tmin, max(tspace))


    tx_depths = bscan_sim.tx_depths
    tx_ranges = np.zeros(len(tx_depths))
    zprof_sim = bscan_sim.z_profile.real
    ax2.set_title('Antenna Positions')
    ax2.scatter(tx_ranges, tx_depths, c='k')
    rxList = bscan_sim.rxList
    for i in range(len(rxList)):
        if i == 0:
            ax2.scatter(rxList[i].x, rxList[i].z, c='g',label='Transmitters TX')
        else:
            ax2.scatter(rxList[i].x, rxList[i].z, c='g')
    ax2.grid()
    ax2.set_ylim(max(zprof_sim) + 1, -1)
    ax2.set_xlabel('Range x [m]')
    ax2.set_ylabel('Depth z [m]')

    x_list = [0, x_rx]
    z_list = [z_tx, z_rx]
    ax2.plot(x_list, z_list, c='k',label='Receivers RX')
    ax2.legend()
    if path2plot != None:
        if os.path.isdir(path2plot) == False:
            os.system('mkdir ' + path2plot)
        fname_ascan = path2plot + '/'
        fname_ascan += 'ascan_compare_z_tx=' + str(z_tx) + '_x_rx=' + str(x_rx) + '_z_rx=' + str(z_rx) + '_2fig_plot.png'
        fig.savefig(fname_ascan)
    pl.show()

def compare_ascans3(bscan_data, bscan_sim, z_tx, x_rx, z_rx, tmin=None, tmax=None, mode_plot = 'pulse', path2plot=None):
    ascan_sim = bscan_sim.get_ascan_from_depth(z_tx=z_tx, x_rx=x_rx, z_rx=z_rx)
    ascan_data = bscan_data.get_ascan_from_depth(z_tx=z_tx, x_rx=x_rx, z_rx=z_rx)
    tx_sig = bscan_sim.tx_signal
    tspace = bscan_data.tspace

    fig = pl.figure(figsize=(20,8),dpi=120)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    fig_label = '$f_{central}$ = ' + str(tx_sig.frequency*1e3) + ' MHz B = ' + str(tx_sig.bandwidth*1e3) + ' MHz \n'
    fig_label += '$z_{tx} = $ ' + str(z_tx) + ' m, $R = $ ' + str(x_rx) + ' m $z_{rx} = $ ' + str(z_rx) + ' m'
    m_ij = misfit_function_ij(ascan_data, ascan_sim, tspace,mode='Waveform')
    # m_ij0 = misfit_function_ij(ascan_data, np.roll(ascan_data, 3), tspace)
    # print(m_ij0)
    fig_label += '\n$m_{ij} =$ ' + str(round(m_ij, 2)) + '$s_{ij} = $' + str(round(1 / m_ij, 5))
    ax1.set_title(fig_label)

    parent_dir1 = os.path.dirname(bscan_sim.fname)
    parent_dir2 = os.path.dirname(bscan_data.fname)
    N_str1 = len(parent_dir1)
    N_str2 = len(parent_dir2)
    label_sim = 'Simulation ' + bscan_sim.fname[N_str1 + 1:]
    label_data = 'PseudoData ' + bscan_data.fname[N_str2 + 1:]

    if mode_plot == 'pulse':
        ax1.plot(tspace, ascan_data.real, label=label_data,c='r')
        ax1.plot(tspace, ascan_sim.real, label=label_sim,c='b',alpha=0.8)
    elif mode_plot == 'abs':
        ax1.plot(tspace, abs(ascan_data), label=label_data,c='r')
        ax1.plot(tspace, abs(ascan_sim), label=label_sim,c='b',alpha=0.8)
    elif mode_plot == 'correlation':
        sig_tx = bscan_sim.tx_signal.pulse
        sig_sim_correl = correlate(ascan_sim, sig_tx)
        sig_data_correl = correlate(ascan_data, sig_tx)
        tspace_lag = np.linspace(-max(tspace), max(tspace), len(sig_sim_correl))
        j_cut = util.findNearest(tspace_lag, 0)
        tspace_cut = tspace_lag[j_cut:]
        sig_sim_correl = abs(sig_sim_correl[j_cut:])
        sig_data_correl = abs(sig_data_correl[j_cut:])
        ax1.plot(tspace_cut, sig_data_correl, c='r', label=label_data)
        ax1.plot(tspace_cut, sig_sim_correl, c='b', label=label_sim,alpha=0.8)
    elif mode_plot == 'envelope':
        E_data = np.sqrt(ascan_data.real**2 + ascan_data.imag**2)
        E_sim = np.sqrt(ascan_sim.real**2 + ascan_sim.imag**2)
        ax1.plot(tspace, E_data, label=label_data, c='r')
        ax1.plot(tspace, E_sim, label=label_sim, c='b', alpha=0.8)

    ax1.grid()
    ax1.legend()
    ax1.set_ylabel('Amplitude A [u]')
    ax1.set_xlabel('Time t [ns]')
    if tmin != None and tmax != None:
        ax1.set_xlim(tmin, tmax)
    elif tmin == None and tmax != None:
        ax1.set_xlim(min(tspace), tmax)
    elif tmin != None and tmax == None:
        ax1.set_xlim(tmin, max(tspace))

    nprof_sim = bscan_sim.n_profile
    zprof_sim = bscan_sim.z_profile.real
    nprof_data = bscan_data.n_profile
    zprof_data = bscan_data.z_profile.real
    ax2.plot(nprof_sim, zprof_sim,c='b')
    ax2.plot(nprof_data, zprof_data,c='r')
    ax2.axhline(0,c='k')
    ax2.grid()
    ax2.set_xlim(1.2,1.8)
    ax2.set_ylim(max(zprof_sim)+1,-1)
    ax2.set_xlabel('Ref Index')
    ax2.set_ylabel('Depth Z [m]')

    tx_depths = bscan_sim.tx_depths
    tx_ranges = np.zeros(len(tx_depths))
    ax3.scatter(tx_ranges, tx_depths, c='k')
    rxList = bscan_sim.rxList
    for i in range(len(rxList)):
        ax3.scatter(rxList[i].x, rxList[i].z,c='g')
    ax3.grid()
    ax3.set_ylim(max(zprof_sim)+1, -1)
    ax3.set_xlabel('Range x [m]')
    ax3.set_ylabel('Depth z [m]')

    x_list = [0, x_rx]
    z_list = [z_tx, z_rx]
    ax3.plot(x_list, z_list, c='k')

    if path2plot != None:
        if os.path.isdir(path2plot) == False:
            os.system('mkdir ' + path2plot)
        fname_ascan = path2plot + '/'
        fname_ascan += 'ascan_compare_z_tx=' + str(z_tx) + '_x_rx=' + str(x_rx) + '_z_rx=' + str(z_rx) + '_3fig_plot.png'
        fig.savefig(fname_ascan)
    pl.show()