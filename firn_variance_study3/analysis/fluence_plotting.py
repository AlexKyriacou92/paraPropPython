import numpy as np
import math
import h5py
import sys
from sys import argv, exit
import configparser

import peakutils
from mpi4py import MPI
import meep as mp
import os
from numpy import array
from matplotlib.colors import LogNorm

from os.path import join
from matplotlib import pyplot as pl
from scipy.signal import butter, lfilter
import utils_CFM
import util_nuRadioMC

from pathlib import Path
import utils_CFM
#from utils_CFM import fluence, get_2peaks, shift_t

sys.path.append('../../')
from paraPropPython import paraProp as ppp
import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rxList_from_file, create_tx_signal, create_transmitter_array
from data import create_hdf_bscan, bscan_rxList, create_hdf_FT
import util
from util import cut_arr2, cut_arr, findNearest
from data import ascan

from fluence_analysis import *

def plot_fluence_map(rxPulses_l, rxList, sourceDepth, ii_select=0, var_mode=True,
                     ppp_mode=True, R_mode=True, tot_mode = False,
                     log_mode=True, show_mode = True, map_cut = [100, 290, None, 170],
                     fname_out = None, path2plots = '', vmin=None, vmax=None, eV_mode=False,
                     cmap='viridis', interp_mode='sinc', title_suffix = None,
                     figsize =(12, 8),fontsize=16, labelsize=12):
    if var_mode == True:
        if tot_mode == True:
            fluence_w = rxPulses_l
        else:
            if R_mode == True:
                fluence_w = rxPulses_l[:,1]
            else:
                fluence_w = rxPulses_l[:,0]
    else:
        if tot_mode == True:
            print('shape of rxPulses', rxPulses_l.shape)
            fluence_w = rxPulses_l[ii_select,:]
        else:
            if R_mode == True:
                fluence_w = rxPulses_l[ii_select, :, 1]
            else:
                fluence_w = rxPulses_l[ii_select, :, 0]
    x_rx_arr = rxList[:,0]
    x_rx_un = np.unique(x_rx_arr)
    z_rx_arr = rxList[:,1]
    z_rx_un = np.unique(z_rx_arr)
    nBins_x = len(x_rx_un)
    nBins_z = len(z_rx_un)

    print('tot_mode', tot_mode, 'R_mode', R_mode, 'fleunce_w shape', fluence_w.shape)

    hist2d, x_bins, z_bins = np.histogram2d(x_rx_arr, z_rx_arr,
                                            bins=(nBins_x, nBins_z),
                                            weights=fluence_w)

    hist2d = np.transpose(hist2d)
    x_cent = (x_bins[1:] + x_bins[:-1]) / 2.
    z_cent = (z_bins[1:] + z_bins[:-1]) / 2.

    X_max = max(x_bins)
    X_min = min(x_bins)
    Z_max = max(z_bins)
    Z_min = min(z_bins)
    map_ranges = [X_min, X_max, Z_min, Z_max]
    plot_ranges = []
    for k in range(len(map_cut)):
        if map_cut[k] == None:
            plot_ranges.append(map_ranges[k])
        else:
            plot_ranges.append(map_cut[k])
    if var_mode == False:
        if tot_mode == True:
            z_symbol = '$\phi^{E}_{tot}$'
            title_prefix = 'Total (tot) Fluence'
        else:
            if R_mode == True:
                z_symbol = '$\phi^{E}_{R}$'
                title_prefix = 'Secondary (R) Fluence '
            else:
                z_symbol = '$\phi^{E}_{D}$'
                title_prefix = 'Primary (D) Fluence '
    else:
        if tot_mode == True:
            z_symbol = '$\Delta \phi^{E}_{tot}/\phi^{E}_{tot}$'
            title_prefix = 'Total (tot) Fluence Variance '
        else:
            if R_mode == True:
                z_symbol = '$\Delta \phi^{E}_{R}/\phi^{E}_{R}$'
                title_prefix = 'Secondary (R) Fluence Variance '
            else:
                z_symbol = '$\Delta \phi^{E}_{D}/\phi^{E}_{D}$'
                title_prefix = 'Primary (D) Fluence Variance '
    title_pl = title_prefix + str(z_symbol) + ', $z_{tx} = ' + str(sourceDepth) + '\,  \mathrm{m}$'
    if eV_mode == True:
        z_symbol += ' [$\mathrm{eV/m^{2}}$]'
    if ppp_mode == True:
        title_pl += ' (paraProp)'
    else:
        title_pl += ' (Meep)'
    if title_suffix != None:
        title_pl += ' ' + title_suffix

    fig = pl.figure(figsize=figsize, dpi=120)
    ax = fig.add_subplot(111)
    ax.set_title(title_pl, fontsize=fontsize)
    if log_mode == True:
        pmesh = ax.imshow(hist2d, extent=[X_min, X_max, Z_max, Z_min], aspect='auto',
                          interpolation=interp_mode, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
    else:
        pmesh = ax.imshow(hist2d, extent=[X_min, X_max, Z_max, Z_min], aspect='auto',
                          interpolation=interp_mode, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(pmesh)
    cbar.set_label(z_symbol, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=labelsize)

    ax.set_aspect(1)
    ax.set_xlabel('Range $x_{rx}$ [m]', fontsize=fontsize)
    ax.set_ylabel('Depth $z_{rx}$ [m]', fontsize=fontsize)
    ax.set_xlim(plot_ranges[0], plot_ranges[1])
    ax.set_ylim(plot_ranges[3], plot_ranges[2])
    ax.tick_params(axis='both', labelsize=labelsize)
    #print('plot complete')
    if fname_out != None:
        if len(path2plots) > 0:
            fname_img = join(path2plots, fname_out)
            nDir = len(path2plots.split('/'))
            if nDir == 1:
                if os.path.isdir(path2plots) == False:
                    os.system('mkdir ' + path2plots)
            else:
                path_l = path2plots.split('/')
                dir_accum = ''
                for k in range(nDir):
                    dir_accum += path_l[k]
                    if os.path.isdir(dir_accum) == False:
                        os.system('mkdir ' + path2plots)
                fname_img = join(path2plots, fname_out)
        else:
            fname_img = fname_out
        fig.savefig(fname_img, bbox_inches='tight')
    if show_mode == True:
        pl.show()
    else:
        pl.close(fig)

import matplotlib.patches as mpatches
def add_label(violin, label, labels):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))

def get_fluence_at_depth(rxPulses_l, rxList, sourceDepth, z_rx_l, color_list, error_mode=False,
                         ppp_mode=True, R_mode=True, tot_mode = False, vio_mode = False,
                         log_mode=True, show_mode = True, title_suffix='', title_prefix='',
                         figsize=(8,5), fontsize=16, labelsize=12):
    '''
    This plots the fluences of the list as a function a receiver depth
    Creates a violin plot
    -if error_mode == True, then it displays the relative residuals of the fluence from the average for each range
    -else: it displays the average fleunce with the errorbars representing the residual errors
    '''
    nSims = len(rxPulses_l)
    nRx_in = len(z_rx_l)
    #nRx = len(rxList)
    x_rx_all = rxList[:,0]
    z_rx_all = rxList[:,1]
    x_rx_un = np.unique(x_rx_all)
    nRx_x = len(x_rx_un)
    ii_x_arr = np.zeros((nRx_in, nRx_x)) #list of indices corresponding to z_rx


    z_rx_un = np.unique(z_rx_all)
    for j in range(nRx_in):
        z_rx = z_rx_l[j]
        ii_z_rx = util.findNearest(z_rx_un, z_rx)
        z_rx_nearest = z_rx_un[ii_z_rx]
        for i in range(nRx_x):
            ii_rx = utils_CFM.get_index_from_rxList(rxList, x_rx_un[i], z_rx_nearest)
            ii_x_arr[j,i] = ii_rx

    rxFluence_at_x = np.zeros((nRx_in, nRx_x, nSims))
    if error_mode == False:
        for k in range(nRx_in):
            for i in range(nRx_x):
                ii_rx = int(ii_x_arr[k, i])
                for j in range(nSims):
                    if tot_mode == True:
                        rxFluence_at_x[k,i,j] = rxPulses_l[j][ii_rx]
                    else:
                        if R_mode == True:
                            rxFluence_at_x[k, i, j] = rxPulses_l[j][ii_rx,1]
                        else:
                            rxFluence_at_x[k, i, j] = rxPulses_l[j][ii_rx,0]
    else:
        for k in range(nRx_in):
            for i in range(nRx_x):
                ii_rx = int(ii_x_arr[k,i])
                fluence_at_x = np.zeros(nSims)
                for j in range(nSims):
                    if tot_mode == True:
                        fluence_at_x[j] = rxPulses_l[j][ii_rx]
                    else:
                        if R_mode == True:
                            fluence_at_x[j] = rxPulses_l[j][ii_rx,1]
                        else:
                            fluence_at_x[j] = rxPulses_l[j][ii_rx,0]
                        #fluence_at_x[j] = rxFluence_at_x[i, j]
                fluence_mean = np.nanmean(fluence_at_x)
                residual_at_x = abs(fluence_at_x - fluence_mean)/fluence_mean
                rxFluence_at_x[k, i,:] = residual_at_x
    fig = pl.figure(figsize=figsize, dpi=120)
    ax = fig.add_subplot(111)
    labels = []


    for k in range(nRx_in):
        rxFluence_violin = []
        rxFluence_mean = []
        rxFluence_std = []

        for i in range(nRx_x):
            print(rxFluence_at_x[k,i])
            rxFluence_violin.append(rxFluence_at_x[k,i])
            rxFluence_mean.append(np.mean(rxFluence_at_x[k,i]))
            rxFluence_std.append(np.std(rxFluence_at_x[k,i]))

        label_k = '$z_{rx} = $' + str(z_rx_l[k]) + ' m'
        if vio_mode == True:
            vio = ax.violinplot(dataset=rxFluence_violin, positions=x_rx_un, showmeans=True,widths=4)
            for partname in ('cmeans', 'cmins', 'cmaxes', 'cbars'):
                vp_line = vio.get(partname)
                if vp_line is not None:
                    if isinstance(vp_line, list):
                        for line in vp_line:
                            line.set_color(color_list[k])
                    else:
                        vp_line.set_color(color_list[k])
            for pc in vio['bodies']:
                pc.set_facecolor(color_list[k])
                pc.set_edgecolor(color_list[k])
            add_label(vio, label_k,labels)
        else:
            ax.errorbar(x_rx_un, rxFluence_mean, rxFluence_std, fmt='o', label=label_k, c=color_list[k])

    ax.set_xlabel('Range $x_{rx}$ [m]', fontsize=fontsize)
    if tot_mode == True:
        low_symbol = 'tot'
    else:
        if R_mode == True:
            low_symbol = 'R'
        else:
            low_symbol = 'D'
    ax.set_yscale('log')
    if error_mode == True:
        y_symbol = '$\Delta \phi_{' + low_symbol + '}/\phi_{' + low_symbol + '}$'
    else:
        y_symbol = '$\phi_{' + low_symbol + '}$ [$\mathrm{eV/m^{2}}$]'
    ax.set_ylabel(y_symbol,fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=labelsize)
    if vio_mode == True:
        ax.legend(*zip(*labels), fontsize=fontsize)
    else:
        ax.legend(fontsize=fontsize)
    title_str = ''
    if title_prefix != None:
        title_str += title_prefix
    if error_mode == True:
        title_str += 'Fluence (' + low_symbol + ') Resiudal ' + y_symbol + ' Distribution'
    else:
        title_str += 'Fluence (' + low_symbol + ') ' + y_symbol + ' Distribution'
    title_str += '$z_{tx} = $' + str(sourceDepth) + ' m'
    if title_suffix != None:
        title_str += title_suffix
    ax.set_title(title_str, fontsize=fontsize)
    ax.grid()
    if show_mode == True:
        pl.show()
    else:
        pl.close(fig)

def plot_pulse_list(ppp_data_in, meep_data_in,
                 x_rx_l, z_rx_l, ii_select = 0, fontsize=16, labelsize=12,
                 fname_pl = None, path2plot = None, title_suffix=None,
                 figsize=(16, 12), title_str=None, show_mode=False, print_mode=False,
                 scale_mode=False, align_mode=False, abs_mode=False, const_yscale=False):

    nRows = len(z_rx_l)
    nCols = len(x_rx_l)

    color_ppp = 'purple'
    color_meep = 'g'

    ascan_l_ppp, tspace_ppp, rxList_ppp, sourceDepth_ppp = ppp_data_in
    rxPulses_l_meep, tspace_meep0, rxList_meep, sourceDepth_meep = meep_data_in

    ascan_0 = ascan_l_ppp[ii_select]
    rxPulses0 = rxPulses_l_meep[ii_select]
    if len(tspace_meep0) == len(rxPulses0[0]):
        tspace_meep = tspace_meep0
    else:
        nSamples_meep = len(rxPulses0[0])
        dt_meep = tspace_meep0[1]-tspace_meep0[0]
        tspace_meep = np.linspace(0, dt_meep*nSamples_meep, nSamples_meep)

    fig, axes = pl.subplots(nrows=nRows, ncols=nCols,
                            figsize=figsize,dpi=120)
    if title_str == None:
        subpl_str = 'Pulse traces (Meep & paraProp), $z_{tx} = $' + str(sourceDepth_ppp)
    else:
        subpl_str = title_str + ', pulse traces (Meep & paraProp), $z_{tx} = $' + str(sourceDepth_ppp)

    if title_suffix != None:
        subpl_str += title_suffix + '\n'
    else:
        subpl_str += '\n'

    if abs_mode == True:
        subpl_str += 'Abs Mag., '
    else:
        subpl_str += 'Real Mag., '
    if align_mode == True:
        subpl_str += '$t_{D}$ alignment enforced, '
    else:
        subpl_str +=  '$t_{D}$ alignment NOT enforced, '
    if scale_mode == True:
        subpl_str += 'amplitudes scaled'
    else:
        subpl_str += 'amplitudes NOT scaled'

    fig.suptitle(subpl_str, fontsize=fontsize)
    t1_l = []
    t2_l = []
    amp_m_l = []
    for i in range(nRows):
        for j in range(nCols):
            if nRows == 1:
                ax1 = axes[j]
            elif nCols == 1:
                ax1 = axes[i]
            else:
                ax1 = axes[i, j]

            x_ex = x_rx_l[j]
            z_ex = z_rx_l[i]

            #ii_ppp = util.get_rx_id(x_ex, z_ex, rxList_ppp)
            ii_ppp = utils_CFM.get_index_from_rxList(rxList_ppp, x_ex, z_ex)
            rx_ppp = rxList_ppp[ii_ppp]
            x_ex_ppp = rx_ppp[0]
            z_ex_ppp = rx_ppp[1]

            rx_pulse_ppp = -1*ascan_0.get_ascan(sourceDepth_ppp, x_ex_ppp, z_ex_ppp)

            ii_rx_meep = utils_CFM.get_index_from_rxList(rxList_meep, x_ex, z_ex)
            rx_meep = rxList_meep[ii_rx_meep]
            x_ex_meep = rx_meep[0]
            z_ex_meep = rx_meep[1]

            delta_z = -1*(z_ex_ppp - sourceDepth_ppp)
            delta_x = x_ex_ppp
            theta_rx = np.arctan(delta_z/delta_x)


            rx_pulse_meep = rxPulses0[ii_rx_meep]
            rx_pulse_meep_env = util.hilbertEnvelope(rx_pulse_meep.real)

            ii_max_meep = np.argmax(abs(rx_pulse_meep))
            t_max_meep = tspace_meep[ii_max_meep]
            amp_max_meep = abs(rx_pulse_meep[ii_max_meep])

            ii_max_ppp = np.argmax(abs(rx_pulse_ppp))
            t_max_ppp = tspace_ppp[ii_max_ppp]
            amp_max_ppp = abs(rx_pulse_ppp[ii_max_ppp])

            amp_ratio = amp_max_meep/amp_max_ppp

            amp_m_l.append(abs(rx_pulse_meep[ii_max_meep]))
            amp_m_l.append(abs(rx_pulse_ppp[ii_max_ppp]))

            dt_ppp = ascan_0.dt

            delta_t_mp = t_max_meep - t_max_ppp
            delta_ii_ppp = int(delta_t_mp/dt_ppp)
            if print_mode == True:
                print(title_str)
                print('Abs_mode?', abs_mode, 'Align mode?', align_mode, 'Scale mode?', scale_mode)
                print('TX', sourceDepth_ppp,'RX', x_ex_ppp, z_ex_ppp)
                print('x_rx match?', x_ex_meep == x_ex_ppp, 'z_rx match?', z_ex_meep == z_ex_ppp)
                print('z_tx match?', sourceDepth_ppp == sourceDepth_meep)
                print('TX', sourceDepth_ppp,'RX', x_ex_ppp, z_ex_ppp)
                print('theta = ', np.rad2deg(theta_rx))
                print('amp_ratio = ampMax_meep/ampMax_ppp =', amp_ratio)
                print('delta_t = tMax_meep - tMax_ppp =', delta_t_mp, 'ns')
                print('abs_t_ratio = tMax_meep/tMax_ppp', t_max_meep/t_max_ppp)
                print('abs_t_ratio (inv) = tMax_ppp/tMax_meep', t_max_ppp/t_max_meep)
                print('')

            if abs_mode == True:
                rx_pulse_meep_pl = abs(rx_pulse_meep_env)
                rx_pulse_ppp_pl = abs(rx_pulse_ppp)
            else:
                rx_pulse_ppp_pl = rx_pulse_ppp.real
                rx_pulse_meep_pl = rx_pulse_meep.real
            if align_mode == True:
                rx_pulse_ppp_pl = np.roll(rx_pulse_ppp_pl, delta_ii_ppp)
            if scale_mode == True:
                rx_pulse_ppp_pl = amp_ratio * rx_pulse_ppp_pl

            ax1.plot(tspace_ppp, rx_pulse_ppp_pl, c=color_ppp,label='ppp',alpha=0.7)
            ax1.plot(tspace_meep, rx_pulse_meep_pl, c=color_meep,label='meep',alpha=0.7)

            ax1.set_title('rx (x = ' + str(x_ex_ppp) + ' m, z = ' + str(z_ex_ppp) + ' m)',fontsize=fontsize)
            ax1.grid()

            if i == nRows-1:
                ax1.set_xlabel('Time t [ns]',fontsize=fontsize)
            if j == 0:
                ax1.set_ylabel('Amplitude V [V/m]',fontsize=fontsize)
            #ax1.ticklabel_format(style='sci')
            ax1.tick_params(axis='both', labelsize=labelsize)
            ax1.legend(fontsize=labelsize)

            t1_meep, t2_meep = get_t1_t2(pulse=rx_pulse_meep_pl, t=tspace_meep)
            t1_ppp, t2_ppp = get_t1_t2(pulse=rx_pulse_ppp_pl, t=tspace_ppp)
            t1_l.append(t1_meep)
            t1_l.append(t1_ppp)
            t2_l.append(t2_meep)
            t2_l.append(t2_ppp)
    t_low = np.min(t1_l)
    t_high = np.max(t2_l)

    amp_high = np.max(amp_m_l)

    for i in range(nRows):
        for j in range(nCols):
            ax1.set_xlim(t_low, t_high)
            if const_yscale == True:
                if abs_mode == False:
                    ax1.set_ylim(-amp_high, amp_high)
                else:
                    axes[i, j].set_ylim(0, 1.2*amp_high)
    if fname_pl != None:
        if path2plot == None:
            fname_img = fname_pl
        else:
            nDir = len(path2plot.split('/'))
            if nDir == 1:
                if os.path.isdir(path2plot) == False:
                    os.system('mkdir ' + path2plot)
            else:
                path_l = path2plot.split('/')
                dir_accum = ''
                for k in range(nDir):
                    if k == 0:
                        dir_accum += path_l[k]
                    else:
                        dir_accum = join(dir_accum, path_l[k])

                    if os.path.isdir(dir_accum) == False:
                        os.system('mkdir ' + dir_accum)
            fname_img = join(path2plot, fname_pl)
        fig.savefig(fname_img, bbox_inches='tight')
    if show_mode == True:
        pl.show()
    else:
        pl.close(fig)

def plot_pulse_variation(data_in, x_rx_l, z_rx_l, label_l=None, color_l = None, ppp_mode=True, alpha=0.7,
                         fontsize=16, labelsize=12,  fname_pl = None, path2plot = None, title_suffix=None,
                         figsize=(16, 12), title_str=None, show_mode=False, print_mode=False, abs_mode=False):
    nRows = len(z_rx_l)
    nCols = len(x_rx_l)
    if ppp_mode == True:
        ascan_l, tspace0, rxList, sourceDepth = data_in
        rxPulses_l = []
        for i in range(len(ascan_l)):
            rxPulses_l.append(ascan_l[i].ascan_array)
        rxPulses_l = np.array(rxPulses_l)
    else:
        rxPulses_l, tspace0, rxList, sourceDepth = data_in
    rx_pulse0 = rxPulses_l[0][0]
    if len(tspace0) == len(rx_pulse0):
        tspace = tspace0
    else:
        nSamples = len(rx_pulse0)
        dt = abs(tspace0[1]-tspace0[0])
        tspace = np.linspace(0, dt*nSamples, nSamples)
    nSims = len(rxPulses_l)

    fig, axes = pl.subplots(nrows=nRows, ncols=nCols,
                            figsize=figsize, dpi=120)
    if title_str == None:
        if ppp_mode == True:
            subpl_str = 'Pulse traces (paraProp) $z_{tx} = $' + str(sourceDepth)
        else:
            subpl_str = 'Pulse traces (Meep) $z_{tx} = $' + str(sourceDepth)

    else:
        if ppp_mode == True:
            subpl_str = title_str + ', pulse traces (paraProp) $z_{tx} = $' + str(sourceDepth)
        else:
            subpl_str = title_str + ', pulse traces (Meep) $z_{tx} = $' + str(sourceDepth)
    if abs_mode == True:
        subpl_str += ' m, Abs Mag. '
    else:
        subpl_str += ' m, Real Mag. '
    if title_suffix != None:
        subpl_str += title_suffix

    fig.suptitle(subpl_str, fontsize=fontsize)

    #TODO: Set Print Mode
    for i in range(nRows):
        for j in range(nCols):
            if nRows == 1:
                ax1 = axes[j]
            elif nCols == 1:
                ax1 = axes[i]
            else:
                ax1 = axes[i, j]

            x_ex = x_rx_l[j]
            z_ex = z_rx_l[i]

            ii_rx = utils_CFM.get_index_from_rxList(rxList, x_ex, z_ex)
            rx_actual = rxList[ii_rx]
            x_ex_actual = rx_actual[0]
            z_ex_actual = rx_actual[1]
            t_low_l = []
            t_high_l = []
            amp_low_l = []
            amp_high_l = []
            for k in range(nSims):
                rx_pulse_k = rxPulses_l[k][ii_rx]
                rx_pulse_re = rx_pulse_k.real
                rx_pulse_im = util.hilbertTransform(rx_pulse_re)
                rx_pulse_c = rx_pulse_re + 1j*rx_pulse_im
                print(len(tspace))
                t_low_k, t_high_k = get_t1_t2(pulse=rx_pulse_c, t=tspace, f1=0.01, f2=0.8, t_tol=50)


                pulse_abs_cut = cut_arr(abs(rx_pulse_c), tspace, t_low_k, t_high_k)
                t_cut = cut_arr(tspace, tspace, t_low_k, t_high_k)
                indices = peakutils.indexes(pulse_abs_cut, thres=0.2)
                t_peaks = peakutils.interpolate(t_cut, pulse_abs_cut, indices)
                amp_peaks = pulse_abs_cut[indices]
                nPeaks = len(indices)
                print('nPeaks = ', nPeaks)
                if nPeaks >= 2:
                    tR = t_peaks[1]
                    t_max_k = tR + 15
                    t_min_k = tR - 15
                    amp_min = -1.25*amp_peaks[1]
                    amp_max = 1.25*amp_peaks[1]
                #TODO: Make else statement
                '''
                else:
                    t_max_k = t_high_k + 25
                    t_min_k = t_high_k - 25
                '''

                t_high_l.append(t_max_k)
                t_low_l.append(t_min_k)
                amp_low_l.append(amp_min)
                amp_high_l.append(amp_max)


                if label_l == None:
                    if color_l == None:
                        ax1.plot(tspace, rx_pulse_c, alpha=alpha)
                    else:
                        ax1.plot(tspace, rx_pulse_c, c=color_l[k],alpha=alpha)
                else:
                    if color_l == None:
                        ax1.plot(tspace, rx_pulse_c, label=label_l[k], alpha=alpha)
                    else:
                        ax1.plot(tspace, rx_pulse_c, label=label_l[k], c=color_l[k], alpha=alpha)
            ax1.set_title('rx (x = ' + str(x_ex_actual) + ' m, z = ' + str(z_ex_actual) + ' m)', fontsize=fontsize)
            ax1.grid()
            # TODO: Set xlim():
            # TODO: Set D and R modes
            t_low = np.min(t_low_l)
            t_high = np.max(t_high_l)
            amp_low = np.min(amp_low_l)
            amp_high = np.max(amp_high_l)
            ax1.set_xlim(t_low, t_high)
            ax1.set_ylim(amp_low, amp_high)

            if i == nRows - 1:
                ax1.set_xlabel('Time t [ns]', fontsize=fontsize)
            if j == 0:
                ax1.set_ylabel('Amplitude V [V/m]', fontsize=fontsize)
            ax1.tick_params(axis='both', labelsize=labelsize)
            ax1.legend(fontsize=labelsize)

    if fname_pl != None:
        if path2plot == None:
            fname_img = fname_pl
        else:
            nDir = len(path2plot.split('/'))
            if nDir == 1:
                if os.path.isdir(path2plot) == False:
                    os.system('mkdir ' + path2plot)
            else:
                path_l = path2plot.split('/')
                dir_accum = ''
                for k in range(nDir):
                    if k == 0:
                        dir_accum += path_l[k]
                    else:
                        dir_accum = join(dir_accum, path_l[k])
                    if os.path.isdir(dir_accum) == False:
                        os.system('mkdir ' + dir_accum)
            fname_img = join(path2plot, fname_pl)
        fig.savefig(fname_img, bbox_inches='tight')
    if show_mode == True:
        pl.show()
    else:
        pl.close(fig)


def plot_spectral_variation(data_in, x_rx_l, z_rx_l, label_l=None, color_l = None, ppp_mode=True, alpha=0.7,
                         fontsize=16, labelsize=12,  fname_pl = None, path2plot = None, title_suffix=None, fmin=0, fmax=0.5,
                         figsize=(16, 12), title_str=None, show_mode=False, print_mode=False, abs_mode=False):
    nRows = len(z_rx_l)
    nCols = len(x_rx_l)


    if ppp_mode == True:
        ascan_l, tspace0, rxList, sourceDepth = data_in
        rxPulses_l = []
        for i in range(len(ascan_l)):
            rxPulses_l.append(ascan_l[i].ascan_array)
    else:
        rxPulses_l, tspace0, rxList, sourceDepth = data_in
    rx_pulse0 = rxPulses_l[0][0]
    nSamples = len(rx_pulse0)
    dt = abs(tspace0[1] - tspace0[0])
    if len(tspace0) == len(rx_pulse0):
        tspace = tspace0
    else:
        tspace = np.linspace(0, dt*nSamples, nSamples)
    nSims = len(rxPulses_l)


    fspace = np.fft.rfftfreq(nSamples, dt)
    fig, axes = pl.subplots(nrows=nRows, ncols=nCols,
                            figsize=figsize, dpi=120)
    if title_str == None:
        if ppp_mode == True:
            subpl_str = 'Pulse traces (paraProp) $z_{tx} = $' + str(sourceDepth)
        else:
            subpl_str = 'Pulse traces (Meep) $z_{tx} = $' + str(sourceDepth)

    else:
        if ppp_mode == True:
            subpl_str = title_str + ', pulse traces (paraProp) $z_{tx} = $' + str(sourceDepth)
        else:
            subpl_str = title_str + ', pulse traces (Meep) $z_{tx} = $' + str(sourceDepth)
    if abs_mode == True:
        subpl_str += ' m, Abs Mag. '
    else:
        subpl_str += ' m, Real Mag. '
    if title_suffix != None:
        subpl_str += title_suffix

    fig.suptitle(subpl_str, fontsize=fontsize)

    #TODO: Set Print Mode
    for i in range(nRows):
        for j in range(nCols):
            if nRows == 1:
                ax1 = axes[j]
            elif nCols == 1:
                ax1 = axes[i]
            else:
                ax1 = axes[i, j]

            x_ex = x_rx_l[j]
            z_ex = z_rx_l[i]

            ii_rx = utils_CFM.get_index_from_rxList(rxList, x_ex, z_ex)
            rx_actual = rxList[ii_rx]
            x_ex_actual = rx_actual[0]
            z_ex_actual = rx_actual[1]

            for k in range(nSims):
                rx_pulse_k = rxPulses_l[k][ii_rx]
                rx_spectrum_k = np.fft.rfft(rx_pulse_k)
                rx_spectrum_abs = abs(rx_spectrum_k)
                print(len(tspace))

                if label_l == None:
                    if color_l == None:
                        ax1.plot(fspace, rx_spectrum_abs, alpha=alpha)
                    else:
                        ax1.plot(fspace, rx_spectrum_abs, c=color_l[k],alpha=alpha)
                else:
                    if color_l == None:
                        ax1.plot(fspace, rx_spectrum_abs, label=label_l[k], alpha=alpha)
                    else:
                        ax1.plot(fspace, rx_spectrum_abs, label=label_l[k], c=color_l[k], alpha=alpha)
            ax1.set_title('rx (x = ' + str(x_ex_actual) + ' m, z = ' + str(z_ex_actual) + ' m)', fontsize=fontsize)
            ax1.grid()
            # TODO: Set xlim():
            # TODO: Set D and R modes

            ax1.set_xlim(fmin, fmax)

            if i == nRows - 1:
                ax1.set_xlabel('Frequency $f$ [GHz]', fontsize=fontsize)
            if j == 0:
                ax1.set_ylabel('Spectral Amplitude S [V/m/Hz]', fontsize=fontsize)
            ax1.tick_params(axis='both', labelsize=labelsize)
            ax1.legend(fontsize=labelsize)

    if fname_pl != None:
        if path2plot == None:
            fname_img = fname_pl
        else:
            nDir = len(path2plot.split('/'))
            if nDir == 1:
                if os.path.isdir(path2plot) == False:
                    os.system('mkdir ' + path2plot)
            else:
                path_l = path2plot.split('/')
                dir_accum = ''
                for k in range(nDir):
                    if k == 0:
                        dir_accum += path_l[k]
                    else:
                        dir_accum = join(dir_accum, path_l[k])
                    if os.path.isdir(dir_accum) == False:
                        os.system('mkdir ' + dir_accum)
            fname_img = join(path2plot, fname_pl)
        fig.savefig(fname_img, bbox_inches='tight')
    if show_mode == True:
        pl.show()
    else:
        pl.close(fig)