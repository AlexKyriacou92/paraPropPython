import numpy as np
import math
import h5py
import sys
from sys import argv, exit
import configparser

import peakutils
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

'''
This script contains functions for calculating fleunce, fluence variation & assorted plotting functions

See other scripts: i.e. plot_fluence_var_2D_200_compare_2.py for their usage

'''
c_light_mns = 0.3
c_light_ms = c_light_mns*1e9
epsilon0 = 8.85e-12

def shift_t(pulse_rx, t, t_shift):
    dt = abs(t[1] - t[0])
    ii_shift = int(t_shift/dt)
    pulse_shift = np.roll(pulse_rx, ii_shift)
    return pulse_shift


def fluence(pulse_v, tspace, tmin=None, tmax=None):
    pulse_v_cut = pulse_v
    if tmin == None and tmax == None:
        pulse_v_cut = pulse_v
    elif tmin == None and tmax != None:
        pulse_v_cut = cut_arr(pulse_v, tspace, min(tspace), tmax)
    elif tmin != None and tmax == None:
        pulse_v_cut = cut_arr(pulse_v, tspace, tmin, max(tspace))
    elif tmin != None and tmax != None:
        pulse_v_cut = cut_arr(pulse_v, tspace, tmin, tmax)
    #print(tmin, tmax)
    return np.sum(abs(pulse_v_cut)**2)

def fluence_eVm2(pulse_v, tspace, tmin=None, tmax=None,  n_local=1.35):
    '''
    Calculates fluence/integrated signal power in units of eV/m^2
    It simply integrates the E-field waveform over a time interval and scales with permittivity and c
    params:
    -pulse_v: input E-field pulse vector - should be in units of V/m
    -tspace: time vector
    -tmin: lower time bound on integral
    -tmax upper time bound on integral
    -n_local: local refractive index - calculates permittivity
    '''


    if tmin == None and tmax == None:
        pulse_v_cut = pulse_v
        #tspace_cut = tspace
    elif tmin == None and tmax != None:
        pulse_v_cut = cut_arr(pulse_v, tspace, min(tspace), tmax)
        #tspace_cut = cut_arr(tspace, tspace, min(tspace), tmax)
    elif tmin != None and tmax == None:
        pulse_v_cut = cut_arr(pulse_v, tspace, tmin, max(tspace))
        #tspace_cut = cut_arr(tspace, tspace, tmin, max(tspace))
    elif tmin != None and tmax != None:
        pulse_v_cut = cut_arr(pulse_v, tspace, tmin, tmax)
        #tspace_cut = cut_arr(tspace, tspace, tmin, tmax)
    tspace_s = tspace*1e-9 # Convert from ns to s
    epsilon = epsilon0*(n_local**2)
    '''
    tspace_s_cut = tspace_cut*1e-9
    tmin_s = tmin*1e-9
    tmax_s = tmax*1e-9
    '''
    dt_s = tspace_s[1]-tspace_s[0] # time interval in s
    nCut = len(pulse_v_cut)
    phi_integrand = 0
    for i in range(nCut):
        phi_integrand += (abs(pulse_v_cut[i])**2)*dt_s
    phi_out_J = c_light_ms*epsilon*phi_integrand
    J_per_eV = 1.6e-19
    phi_out_eV = phi_out_J/J_per_eV
    #print('phi = ', phi_out_eV)
    return phi_out_eV

def get_2peaks(pulse_rx, tspace, t_cut, t_ratio=2, eV_mode=False, n_local=None, plot_mode=False):
    pulse_rx_abs = abs(pulse_rx)
    ii_1 = np.argmax(pulse_rx_abs)
    #amp_max_D = np.max(pulse_rx_abs)
    t_max_1 = tspace[ii_1]
    t_end = tspace[-1]

    pulse_rx_roll = shift_t(pulse_rx, tspace, -t_max_1)
    pulse_rx_abs_roll = shift_t(pulse_rx_abs, tspace, -t_max_1)

    t_cut_R = t_ratio*t_cut
    pulse_rx_abs_cut, tspace_cut = util.cut_arr2(pulse_rx_abs_roll, tspace, t_cut_R, t_end-t_cut_R)
    ii_max_2 = np.argmax(pulse_rx_abs_cut)
    t_max_roll_2 = tspace_cut[ii_max_2] # + t_max_D
    if plot_mode == True:
        fig = pl.figure(figsize=(8,5),dpi=120)
        ax = fig.add_subplot(111)
        ax.plot(tspace, pulse_rx_abs)
        ax.plot(tspace, pulse_rx_roll)
        ax.plot(tspace_cut, pulse_rx_abs_cut)
        pl.close(fig)

    if t_max_roll_2 > t_max_1:
        delta_t2 = t_end - t_max_roll_2
        t_max_D = t_max_1 - delta_t2
        t_max_R = t_max_1

        ii_max_D = findNearest(tspace, t_max_D)
        ii_max_R = ii_1

    elif t_max_1 == t_max_roll_2:
        t_max_D = t_max_1
        t_max_R = t_max_1
        ii_max_D = ii_1
        ii_max_R = ii_1

    else:
        t_max_D = t_max_1
        t_max_R = t_max_1 + t_max_roll_2
        ii_max_R = findNearest(tspace, t_max_R)
        ii_max_D = ii_1
    if eV_mode == False:
        fluence_D = fluence(pulse_rx, tspace, t_max_D - t_cut, t_max_D + t_cut)
        fluence_R = fluence(pulse_rx, tspace, t_max_R-t_cut, t_max_R+t_cut)
    else:
        if n_local == None:
            fluence_D = fluence_eVm2(pulse_v=pulse_rx, tspace=tspace, tmin=t_max_D - t_cut, tmax=t_max_D + t_cut)
            fluence_R = fluence_eVm2(pulse_v=pulse_rx, tspace=tspace, tmin=t_max_R - t_cut, tmax=t_max_R + t_cut)
        else:
            fluence_D = fluence_eVm2(pulse_v=pulse_rx, tspace=tspace, tmin=t_max_D - t_cut, tmax=t_max_D + t_cut, n_local=n_local)
            fluence_R = fluence_eVm2(pulse_v=pulse_rx, tspace=tspace, tmin=t_max_R - t_cut, tmax=t_max_R + t_cut, n_local=n_local)

    dt_DR = t_max_R - t_max_R
    amp_D = pulse_rx_abs[ii_max_D]
    amp_R = pulse_rx_abs[ii_max_R]
    return [fluence_D, fluence_R, amp_D, amp_R, t_max_D, t_max_R, dt_DR]

def get_outputs(output_l):
    fluence_D, fluence_R = output_l[0], output_l[1]
    amp_D, amp_R = output_l[2], output_l[3]
    t_max_D, t_max_R = output_l[4], output_l[5]
    dt_DR = output_l[6]
    #print('fluence_D', 'fleunce_R', 'amp_D', 'amp_R', 't_max_D', 't_max_R', 'dt_DR')
    #print(output_l)
    return fluence_D, fluence_R, amp_D, amp_R, t_max_D, t_max_R, dt_DR

def get_fname_l(fname_config):
    config = configparser.ConfigParser()
    config.read(fname_config)

    input_config = config['DATAFILES']
    label_config = config['LABELS']

    fname_list = []
    for key in input_config.keys():
        fname_k = input_config[key]
        fname_list.append(fname_k)

    label_list = []
    for key in label_config.keys():
        label_k = label_config[key]
        label_list.append(label_k)
    return fname_list, label_list

def get_sim_meep(fname_config):
    config = configparser.ConfigParser()
    config.read(fname_config)

    input_config = config['DATAFILES']
    label_config = config['LABELS']
    rx_config = config['RX']
    fname_list = []

    fname_mmap_l = []
    pol_mode = 'zpol'
    for key in input_config.keys():
        fname_k = input_config[key]
        fname_list.append(fname_k)
        path2data = os.path.split(fname_k)[0]
        fname_mmap = Path(fname_k).stem
        fname_mmap_l.append(join(path2data, fname_mmap))
    nFiles = len(fname_list)

    label_list = []
    for key in label_config.keys():
        label_k = label_config[key]
        label_list.append(label_k)

    fname_hdf0 = fname_list[0]

    with h5py.File(fname_hdf0, 'r') as hdf_in0:
        for key in hdf_in0.attrs.keys():
            print(key, hdf_in0.attrs[key])
        dt = float(hdf_in0.attrs['dt'])
    # Create Memmaps
    rxList = utils_CFM.get_rxList_from_file(fname_hdf=fname_hdf0)
    #tspace = utils_CFM.get_tspace(fname_hdf=fname_hdf0)
    sourceDepth = utils_CFM.get_sourceDepth_from_file(fname_hdf=fname_hdf0)
    rxPulses_l = []

    for i in range(nFiles):
        fname_mmap = fname_mmap_l[i]
        fname_hdf = fname_list[i]
        if os.path.isfile(fname_mmap) == False:
            rxPulses_i = utils_CFM.save_rxPulses_mmap(fname_hdf, fname_mmap, pol_mode)
            rxPulses_l.append(rxPulses_i)
            print(i, 'save mmap', fname_mmap)
        else:
            rxPulses_i = np.load(fname_mmap, 'r')
            rxPulses_l.append(rxPulses_i)
            print(i, 'load mmap', fname_mmap)
    nSamples = len(rxPulses_l[0][0])
    tspace = np.linspace(0, nSamples*dt, nSamples)

    return rxPulses_l, tspace, rxList, sourceDepth

def get_sim_ppp(fname_config):
    config = configparser.ConfigParser()
    config.read(fname_config)

    input_config = config['DATAFILES']
    label_config = config['LABELS']
    rx_config = config['RX']

    fname_list = []
    for key in input_config.keys():
        fname_k = input_config[key]
        # fname_k = join(path2data, fname_k)
        fname_list.append(fname_k)
    label_list = []
    for key in label_config.keys():
        label_k = label_config[key]
        label_list.append(label_k)
    '''
    x_rx = float(rx_config['X_RX'])
    z_rx = float(rx_config['Z_RX'])
    '''

    nSims = len(fname_list)
    # Select First Ascan in List
    fname_0 = fname_list[0]
    ascan_0 = ascan()
    ascan_0.load_from_hdf(fname_0)
    tspace = ascan_0.tspace
    rxList = ascan_0.rxList
    '''
    t_cut = 15
    t_ratio = 1
    '''

    print(ascan_0.tx_depths)
    sourceDepth = ascan_0.tx_depths[0]

    '''
    nRx = len(rxList)
    x_arr, z_arr = [], []
    for k in range(nRx):
        rx_k = rxList[k]
        x_arr.append(rx_k.x)
        z_arr.append(rx_k.z)
    x_arr = array(x_arr)
    z_arr = array(z_arr)
    '''
    ascan_l = []
    for i in range(nSims):
        ascan_i = ascan()
        ascan_i.load_from_hdf(fname_hdf=fname_list[i])
        ascan_l.append(ascan_i)
    nRx = len(rxList)
    rxArr = np.zeros((nRx,2))
    for i in range(nRx):
        rxArr[i,0] = rxList[i].x
        rxArr[i,1] = rxList[i].z

    return ascan_l, tspace, rxArr, sourceDepth

def show_rxList(rxList, ppp_mode = True):
    nRx = len(rxList)
    if ppp_mode == True:
        x_rx_arr = []
        z_rx_arr = []
        for i in range(nRx):
            x_rx_arr.append(rxList[i].x)
            z_rx_arr.append(rxList[i].z)
    else:
        x_rx_arr = rxList[:,0]
        z_rx_arr = rxList[:,1]
    x_rx_un = np.unique(x_rx_arr)
    z_rx_un = np.unique(z_rx_arr)
    print('ranges, x = ', x_rx_un)
    print('depths, z = ', z_rx_un)

# Formatting rxPulses
def convert_rxPulses(rxPulses_in, tspace, rxList_in, x_limits=[None, None], z_limits=[None,None], t_shift = 50):
    '''
    This function converts the shape of the array rxPulses (an array nRx x nSamples) base on cuts to the receiver
    rxPulses_in: Contains signal amplitude at each receiver

    Warning: designed for meep

    tspace: time vector
    rxList: list of receiver positions {x,z}
    x_limits: cuts to the receiver range (x)
    z_limits: cuts to the receiver depth (z)
    t_shift: Used to roll amplitude vector: to account for the starting time of the pulse
    '''
    dt = tspace[1]-tspace[0]
    ii_shift = int(t_shift/dt)
    nRx = len(rxList_in)
    if (x_limits[0] == None) and (x_limits[1] == None) and (z_limits[0] == None) and (z_limits[1] == None):
        rxPulses_out = np.zeros(nRx, dtype='complex') #= rxPulses_in
        for i in range(nRx):
            rxPulses_out[i] = np.roll(rxPulses_in[i],-ii_shift)
        rxList_out = rxList_in
    else:
        x_rx_un = np.unique(rxList_in[:,0])
        z_rx_un = np.unique(rxList_in[:,1])
        nRx_x = len(x_rx_un)
        nRx_z = len(z_rx_un)

        ii_x, ii_z = [], []
        ii_default = [0,-1]
        for j in range(2):
            if x_limits[j] == None:
                ii_x_cut = ii_default[j]
            else:
                ii_x_cut = findNearest(x_rx_un, x_limits[j])
            if z_limits[j] == None:
                ii_z_cut = ii_default[j]
            else:
                ii_z_cut = findNearest(z_rx_un, z_limits[j])
            ii_x.append(ii_x_cut)
            ii_z.append(ii_z_cut)
        ii_l = []
        rxList_out = []
        rxPulses_out = []
        for i in range(ii_x[0], ii_x[1]):
            for j in range(ii_z[0], ii_z[1]):
                ii_rx = utils_CFM.get_index_from_rxList(rxList_in, x_rx_un[i], z_rx_un[j])
                ii_l.append(ii_rx)
                rxList_out.append(rxList_in[ii_rx])
                rxPulses_out.append(np.roll(rxPulses_in[ii_rx], -ii_shift))
        rxList_out = np.array(rxList_out)
        rxPulses_out = np.array(rxPulses_out)
    return rxPulses_out, rxList_out

# Amp and Time Matrices
def get_fluence_matrix(rxPulse_matrix, tspace_in, t_cut, t_ratio, eV_mode=False, n_matrix=None, ppp_mode=True, tot_mode=False, t_limit_in=[]):
    if ppp_mode == False:
        nRx = len(rxPulse_matrix)
    else:
        nRx = rxPulse_matrix.nRx
    t_limit_arr = np.ones((nRx, 2))
    if len(t_limit_in) > 0:
        t_limit_arr = t_limit_in
    else:
        t_limit_arr[:,1] *= max(tspace_in)
        t_limit_arr[:,0] *= 0.

    if tot_mode == True:
        fluence_arr = np.zeros(nRx)
    else:
        fluence_arr = np.zeros((nRx, 2))
    for i in range(nRx):
        pulse_rx_j = rxPulse_matrix[i]
        #Cut Array
        if len(t_limit_in) > 0:
            t_low = t_limit_arr[i,0]
            t_high = t_limit_arr[i,1]
            pulse_rx_j, tspace = cut_arr2(pulse_rx_j, tspace_in, t_low, t_high)
        else:
            tspace = tspace_in

        if ppp_mode == False:
            pulse_rx_r = pulse_rx_j.real
            pulse_rx_i = util.hilbertTransform(pulse_rx_r)
            pulse_rx_c = pulse_rx_r + 1j * pulse_rx_i
        else:
            pulse_rx_c = pulse_rx_j
        if tot_mode == True:
            if eV_mode == False:
                fluence_arr[i] = fluence(pulse_rx_c, tspace)
            else:
                if n_matrix == None:
                    fluence_arr[i] = fluence_eVm2(pulse_rx_c, tspace)
                else:
                    fluence_arr[i] = fluence_eVm2(pulse_rx_c, tspace, n_local=n_matrix[i])
        else:
            #TODO: Add n_matrix here
            output_l = get_2peaks(pulse_rx=pulse_rx_c,
                                  tspace=tspace,
                                  t_cut=t_cut,
                                  t_ratio=t_ratio,
                                  eV_mode=eV_mode,
                                  plot_mode=False)

            fluence_D, fluence_R, amp_D, amp_R, t_max_D, t_max_R, dt_DR = get_outputs(output_l)
            fluence_arr[i, 0] = fluence_D
            fluence_arr[i, 1] = fluence_R
    return fluence_arr

def get_fluence_matrix_list(rxPulses_l, rxList, tspace, t_cut, t_ratio, eV_mode=False, n_matrix=None, fname_npy=None, override=True, ppp_mode=True, tot_mode=False, xCut=None, t_limit_in=[]):
    nSims = len(rxPulses_l)
    nRx = len(rxList)
    if tot_mode == True:
        fluence_arr = np.zeros((nSims, nRx))
    else:
        fluence_arr = np.zeros((nSims, nRx, 2))
    if fname_npy == None:
        for i in range(nSims):
            if ppp_mode == True:
                rxPulse_matrix = rxPulses_l[i].ascan_array[0,:]
            else:
                rxPulse_matrix = rxPulses_l[i]
            fluence_arr[i] = get_fluence_matrix(rxPulse_matrix, tspace, t_cut, t_ratio,
                                                ppp_mode=ppp_mode,tot_mode=tot_mode, eV_mode=eV_mode, n_matrix=n_matrix, t_limit_in=t_limit_in)
    else:
        file_exist = os.path.isfile(fname_npy)
        if file_exist == False or override == True:
            for i in range(nSims):
                if ppp_mode == True:
                    rxPulse_matrix = rxPulses_l[i].ascan_array[0,:]
                else:
                    rxPulse_matrix = rxPulses_l[i]
                fluence_arr[i] = get_fluence_matrix(rxPulse_matrix, tspace, t_cut, t_ratio,
                                                ppp_mode=ppp_mode,tot_mode=tot_mode, eV_mode=eV_mode, n_matrix=n_matrix, t_limit_in=t_limit_in)
            np.save(fname_npy, fluence_arr)
        else:
            fluence_arr = np.load(fname_npy,'r')
    if xCut != None:
        x_rx_arr = rxList[:,0]
        x_rx_un = np.unique(x_rx_arr)
        ii_nearest = util.findNearest(x_rx_un, xCut)
        xCut2 = x_rx_un[ii_nearest]
        for i in range(len(x_rx_arr)):
            if x_rx_arr[i] == xCut2:
                j_cut = i
                break
        fluence_arr_out = fluence_arr[:, j_cut:]
        rxList_out = rxList[j_cut:]
    else:
        rxList_out = rxList
        fluence_arr_out = fluence_arr
    return fluence_arr_out, rxList_out

def get_tpeak_matrix(rxPulse_matrix, tspace_in, t_cut, t_ratio, eV_mode=False, ppp_mode=True, t_limit_in=[]):
    nRx = len(rxPulse_matrix)
    t_limit_arr = np.ones((nRx, 2))
    if len(t_limit_in) > 0:
        t_limit_arr = t_limit_in
    else:
        t_limit_arr[:, 1] *= max(tspace_in)
        t_limit_arr[:, 0] *= 0.

    time_arr = np.zeros((nRx,2))
    for i in range(nRx):
        pulse_rx_j = rxPulse_matrix[i]

        #Cut Array

        if len(t_limit_in) > 0:
            t_low = t_limit_arr[i,0]
            t_high = t_limit_arr[i,1]
            pulse_rx_j, tspace = cut_arr2(pulse_rx_j, tspace_in, t_low, t_high)
        else:
            tspace = tspace_in

        if ppp_mode == False:
            pulse_rx_r = pulse_rx_j.real
            pulse_rx_i = util.hilbertTransform(pulse_rx_r)
            pulse_rx_c = pulse_rx_r + 1j * pulse_rx_i
        else:
            pulse_rx_c = pulse_rx_j
        output_l = get_2peaks(pulse_rx=pulse_rx_c,
                              tspace=tspace,
                              t_cut=t_cut,
                              t_ratio=t_ratio,
                              eV_mode=eV_mode,
                              plot_mode=False)
        fluence_D, fluence_R, amp_D, amp_R, t_max_D, t_max_R, dt_DR = get_outputs(output_l)
        time_arr[i,0] = t_max_D
        time_arr[i,1] = t_max_R
    return time_arr

def get_tpeak_matrix_l(rxPulses_l, rxList, tspace_in, t_cut, t_ratio, eV_mode=False, ppp_mode=True, xCut=None, t_limit_in=[]):
    nSims = len(rxPulses_l)
    nRx = len(rxList)
    time_arr = np.zeros((nSims, nRx, 2))
    for i in range(nSims):
        if ppp_mode == True:
            rxPulse_matrix = rxPulses_l[i].ascan_array[0, :]
        else:
            rxPulse_matrix = rxPulses_l[i]
        time_arr[i] = get_tpeak_matrix(rxPulse_matrix, tspace_in, t_cut, t_ratio, eV_mode, ppp_mode, t_limit_in)
    if xCut != None:
        x_rx_arr = rxList[:, 0]
        x_rx_un = np.unique(x_rx_arr)
        ii_nearest = util.findNearest(x_rx_un, xCut)
        xCut2 = x_rx_un[ii_nearest]
        for i in range(len(x_rx_arr)):
            if x_rx_arr[i] == xCut2:
                j_cut = i
                break
        time_arr_out = time_arr[:, j_cut:]
        rxList_out = rxList[j_cut:]
    else:
        rxList_out = rxList
        time_arr_out = time_arr
    return time_arr_out, rxList_out

def calculate_variance_matrix(fluence_arr, tot_mode=False, t_mode = False):
    '''
    t_mode: calculating times (not fluence)
    '''
    #nSims = fluence_arr.shape[0]
    nRx = fluence_arr.shape[1]
    if tot_mode == True:
        fluence_var_arr = np.zeros(nRx)
    else:
        fluence_var_arr = np.zeros((nRx,2))
    for i in range(nRx):
        if tot_mode == True:
            if t_mode == False:
                fluence_var_arr[i] = np.nanstd(fluence_arr[:,i])/np.nanmean(fluence_arr[:,i])
            else:
                fluence_var_arr[i] = np.nanstd(fluence_arr[:,i])

        else:
            for j in range(2):
                if t_mode == False:
                    fluence_var_arr[i,j] = np.nanstd(fluence_arr[:,i,j])/np.nanmean(fluence_arr[:,i,j])
                else:
                    fluence_var_arr[i,j] = np.nanstd(fluence_arr[:,i,j])
    return fluence_var_arr



# Plotting Functions
def get_t1_t2(pulse, t, f1=0.1, f2=0.85, t_tol=100):
    '''
    Calculates t1 and t2 cuts for xlim for plotting
    xlim(t1,t2) - to ensure that D and R are both visible in the pulse

    pulse : pulse amplitude vector
    t : time vector
    f1 : what percentage of accumulated ampl. to place low cut
    f2 : what percentage of accumulated ampl. to place high cut
    t_tol: what time should be used as a gap from t(f1) and t(f2)
    '''
    pulse_abs = abs(pulse) # Magnitude Vector
    pulse_cs = np.cumsum(pulse_abs)/np.sum(pulse_abs) # Normalized accumulate sum vector
    ii1 = util.findNearest(pulse_cs, f1) # index of low cut
    ii2 = util.findNearest(pulse_cs, f2) # index of high cut
    t1 = t[ii1] - t_tol
    t2 = t[ii2] + t_tol
    return t1, t2

#TODO: This version of the function crashes when in 'var mode'
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom
from scipy.signal import resample
'''
def plot_fluence_map(rxPulses_l, rxList, sourceDepth, ii_select=0, var_mode=True,
                     ppp_mode=True, R_mode=True, tot_mode=False,
                     log_mode=True, show_mode=True, cl_mode=False, map_cut=[100, 290, None, 170],
                     fname_out=None, path2plots='', vmin=None, vmax=None, eV_mode=False,
                     cmap='viridis', interp_mode='spline', title_suffix=None,
                     figsize=(12, 8), fontsize=16, labelsize=12, interp_factor=20, transparent_nan=True):

    if var_mode:
        fluence_w = rxPulses_l if tot_mode else rxPulses_l[:, 1] if R_mode else rxPulses_l[:, 0]
    else:
        if tot_mode:
            print('shape of rxPulses', rxPulses_l.shape)
            fluence_w = rxPulses_l[ii_select, :]
        else:
            fluence_w = rxPulses_l[ii_select, :, 1] if R_mode else rxPulses_l[ii_select, :, 0]

    x_rx_arr = rxList[:, 0]
    z_rx_arr = rxList[:, 1]
    x_rx_un = np.unique(x_rx_arr)
    z_rx_un = np.unique(z_rx_arr)
    nBins_x = len(x_rx_un)
    nBins_z = len(z_rx_un)

    print('tot_mode', tot_mode, 'R_mode', R_mode, 'fluence_w shape', fluence_w.shape)

    hist2d, x_bins, z_bins = np.histogram2d(x_rx_arr, z_rx_arr,
                                            bins=(nBins_x, nBins_z),
                                            weights=fluence_w)
    hist2d = hist2d.T
    x_cent = (x_bins[1:] + x_bins[:-1]) / 2.
    z_cent = (z_bins[1:] + z_bins[:-1]) / 2.

    X_min, X_max = x_bins[0], x_bins[-1]
    Z_min, Z_max = z_bins[0], z_bins[-1]
    map_ranges = [X_min, X_max, Z_min, Z_max]
    plot_ranges = [map_ranges[i] if map_cut[i] is None else map_cut[i] for i in range(4)]

    if not var_mode:
        if tot_mode:
            z_symbol = '$\phi^{E}_{tot}$'
            title_prefix = 'Total (tot) Fluence'
        else:
            z_symbol = '$\phi^{E}_{R}$' if R_mode else '$\phi^{E}_{D}$'
            title_prefix = 'Secondary (R) Fluence' if R_mode else 'Primary (D) Fluence'
    else:
        if tot_mode:
            z_symbol = '$\Delta \phi^{E}_{tot}/\phi^{E}_{tot}$'
            title_prefix = 'Total (tot) Fluence Variance'
        else:
            z_symbol = '$\Delta \phi^{E}_{R}/\phi^{E}_{R}$' if R_mode else '$\Delta \phi^{E}_{D}/\phi^{E}_{D}$'
            title_prefix = 'Secondary (R) Fluence Variance' if R_mode else 'Primary (D) Fluence Variance'

    if eV_mode:
        z_symbol += ' [$\mathrm{eV/m^{2}}$]'

    title_pl = f"{title_prefix} {z_symbol}, $z_{{tx}} = {sourceDepth:.1f}\, \\mathrm{{m}}$"
    title_pl += ' (paraProp)' if ppp_mode else ' (Meep)'
    if title_suffix:
        title_pl += f" {title_suffix}"

    fig = pl.figure(figsize=figsize, dpi=120)
    ax = fig.add_subplot(111)
    ax.set_title(title_pl, fontsize=fontsize)
    if transparent_nan:
        cmap_obj = pl.get_cmap(cmap)
        cmap_mod = cmap_obj.with_extremes(bad='none')  # or bad='white' if you prefer white fill for NaNs

    # Interpolation modes with edge-aware handling
    if interp_mode == 'bilinear':
        interp_func = RegularGridInterpolator((z_cent, x_cent), hist2d, bounds_error=False, fill_value=np.nan)
        x_fine = np.linspace(x_cent[0], x_cent[-1], interp_factor * len(x_cent))
        z_fine = np.linspace(z_cent[0], z_cent[-1], interp_factor * len(z_cent))
        X_plot, Z_plot = np.meshgrid(x_fine, z_fine)
        points = np.array([Z_plot.ravel(), X_plot.ravel()]).T
        hist_plot = interp_func(points).reshape(Z_plot.shape)

    elif interp_mode == 'spline':
        hist_plot = zoom(hist2d, interp_factor, order=3)
        x_fine = np.linspace(x_bins[0], x_bins[-1], hist_plot.shape[1])
        z_fine = np.linspace(z_bins[0], z_bins[-1], hist_plot.shape[0])
        X_plot, Z_plot = np.meshgrid(x_fine, z_fine)

    elif interp_mode == 'sinc':
        hist_temp = resample(hist2d, interp_factor * hist2d.shape[0], axis=0)
        hist_plot = resample(hist_temp, interp_factor * hist2d.shape[1], axis=1)
        x_fine = np.linspace(x_bins[0], x_bins[-1], hist_plot.shape[1])
        z_fine = np.linspace(z_bins[0], z_bins[-1], hist_plot.shape[0])
        X_plot, Z_plot = np.meshgrid(x_fine, z_fine)

    else:  # 'none'
        X_plot, Z_plot = np.meshgrid(x_bins, z_bins)
        hist_plot = hist2d

    # Handle vmin/vmax safely, and clean up invalid values
    if log_mode:
        # Remove zeros and negatives for LogNorm, replace with NaN
        hist_plot = np.where(np.isfinite(hist_plot) & (hist_plot > 0), hist_plot, np.nan)
        valid_mask = ~np.isnan(hist_plot)

        if not np.any(valid_mask):
            raise ValueError("All fluence values are non-positive or invalid — cannot apply LogNorm.")

        if vmin is None:
            vmin = np.nanmin(hist_plot)
        if vmax is None:
            vmax = np.nanmax(hist_plot)
    else:
        # For linear scale, just remove NaNs/infs
        hist_plot = np.where(np.isfinite(hist_plot), hist_plot, np.nan)
        if vmin is None:
            vmin = np.nanmin(hist_plot)
        if vmax is None:
            vmax = np.nanmax(hist_plot)


    if log_mode:
        pmesh = ax.pcolormesh(X_plot, Z_plot, hist_plot, cmap=cmap_mod,
                              norm=LogNorm(vmin=vmin, vmax=vmax), shading='auto')
    else:
        pmesh = ax.pcolormesh(X_plot, Z_plot, hist_plot, cmap=cmap_mod,
                              vmin=vmin, vmax=vmax, shading='auto')

    cbar = fig.colorbar(pmesh)
    cbar.set_label(z_symbol, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=labelsize)

    #ax.set_aspect(1)
    ax.set_xlabel('Range $x_{rx}$ [m]', fontsize=fontsize)
    ax.set_ylabel('Depth $z_{rx}$ [m]', fontsize=fontsize)
    ax.set_xlim(plot_ranges[0], plot_ranges[1])
    ax.set_ylim(plot_ranges[3], plot_ranges[2])
    ax.tick_params(axis='both', labelsize=labelsize)

    if fname_out:
        fname_img = join(path2plots, fname_out) if path2plots else fname_out
        os.makedirs(os.path.dirname(fname_img), exist_ok=True)
        fig.savefig(fname_img, bbox_inches='tight')

    if show_mode:
        pl.show()
    else:
        return pl, ax

'''
def plot_fluence_map(rxPulses_l, rxList, sourceDepth, ii_select=0, var_mode=True,
                     ppp_mode=True, R_mode=True, tot_mode = False,
                     log_mode=True, show_mode = True, cl_mode = False, map_cut = [100, 290, None, 170],
                     fname_out = None, path2plots = '', vmin=None, vmax=None, eV_mode=False,
                     cmap='viridis', interp_mode='spline36', title_suffix = None,
                     figsize =(12, 6),fontsize=16, labelsize=12):
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
    #x_cent = (x_bins[1:] + x_bins[:-1]) / 2.
    #z_cent = (z_bins[1:] + z_bins[:-1]) / 2.

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

    #ax.set_aspect(1)
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
        if cl_mode == True:
            pl.close(fig)
        else:
            return pl, ax

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

    rxPulses_l should actually be rxFluence_l

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
            #print(rxFluence_at_x[k,i])
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
        title_str += 'Fluence (' + low_symbol + ') Residual ' + y_symbol + ' Distribution, '
    else:
        title_str += 'Fluence (' + low_symbol + ') ' + y_symbol + ' Distribution, '
    title_str += '$z_{tx} = $' + str(sourceDepth) + ' m'
    if title_suffix != None:
        title_str += title_suffix
    ax.set_title(title_str, fontsize=fontsize)
    ax.grid()
    if show_mode == True:
        pl.show()
    else:
        return fig, ax

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

def plot_pulse_variation(data_in, x_rx_l, z_rx_l, t_limit_arr=[], label_l=None, color_l = None, ppp_mode=True, alpha=0.7,
                         fontsize=16, labelsize=12,  fname_pl = None, path2plot = None, title_suffix=None,
                         figsize=(16, 8), title_str=None, show_mode=False, t_lim_auto=True, print_mode=False, abs_mode=False):
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
        tspace1 = tspace0
    else:
        nSamples = len(rx_pulse0)
        dt = abs(tspace0[1]-tspace0[0])
        tspace1 = np.linspace(0, dt*nSamples, nSamples)
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
                rx_pulse_k1 = rxPulses_l[k][ii_rx]
                if len(t_limit_arr) > 0:
                    rx_pulse_k, tspace = cut_arr2(rx_pulse_k1, tspace1, t_limit_arr[ii_rx,0], t_limit_arr[ii_rx,1])
                else:
                    tspace = tspace1
                    rx_pulse_k = rx_pulse_k1

                rx_pulse_re = rx_pulse_k.real
                rx_pulse_im = util.hilbertTransform(rx_pulse_re)
                rx_pulse_c = rx_pulse_re + 1j*rx_pulse_im
                #print(len(tspace))
                t_low_k, t_high_k = get_t1_t2(pulse=rx_pulse_c, t=tspace, f1=0.01, f2=0.8, t_tol=50)


                pulse_abs_cut = cut_arr(abs(rx_pulse_c), tspace, t_low_k, t_high_k)
                t_cut = cut_arr(tspace, tspace, t_low_k, t_high_k)
                indices = peakutils.indexes(pulse_abs_cut, thres=0.2)
                t_peaks = peakutils.interpolate(t_cut, pulse_abs_cut, indices)
                amp_peaks = pulse_abs_cut[indices]
                nPeaks = len(indices)
                #print('nPeaks = ', nPeaks)
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
            if t_lim_auto == True:
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
        return fig, axes

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
                #print(len(tspace))

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
        return fig, axes

def plot_t_map(rxPulses_l, rxList, sourceDepth, ii_select=0, var_mode=True,
               ppp_mode=True, R_mode=True, tot_mode = False,
               log_mode=True, show_mode = True, cl_mode = False, map_cut = [100, 290, None, 170],
               fname_out = None, path2plots = '', vmin=None, vmax=None,
               cmap='viridis', interp_mode='spline36', title_suffix = None,
               figsize =(12, 6),fontsize=16, labelsize=12):
    if var_mode == True:
        if R_mode == True:
            time_w = rxPulses_l[:,1]
        else:
            time_w = rxPulses_l[:,0]
    else:
        if R_mode == True:
            time_w = rxPulses_l[ii_select, :, 1]
        else:
            time_w = rxPulses_l[ii_select, :, 0]
    x_rx_arr = rxList[:,0]
    x_rx_un = np.unique(x_rx_arr)
    z_rx_arr = rxList[:,1]
    z_rx_un = np.unique(z_rx_arr)
    nBins_x = len(x_rx_un)
    nBins_z = len(z_rx_un)

    print('tot_mode', tot_mode, 'R_mode', R_mode, 'fleunce_w shape', time_w.shape)

    hist2d, x_bins, z_bins = np.histogram2d(x_rx_arr, z_rx_arr,
                                            bins=(nBins_x, nBins_z),
                                            weights=time_w)

    hist2d = np.transpose(hist2d)

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
        if R_mode == True:
            z_symbol = '$t_{R}$'
            title_prefix = 'Secondary (R) prop. time '
        else:
            z_symbol = '$t_{D}$'
            title_prefix = 'Primary (D) prop. time '
    else:
        if R_mode == True:
            z_symbol = '$\Delta t_{R}$'
            title_prefix = 'Secondary (R) prop. time variance '
        else:
            z_symbol = '$\Delta t_{D}$'
            title_prefix = 'Primary (D) prop. time variance '
    title_pl = title_prefix + str(z_symbol) + ', $z_{tx} = ' + str(sourceDepth) + '\,  \mathrm{m}$'
    z_symbol += ' [ns]'
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

    #ax.set_aspect(1)
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
        if cl_mode == True:
            pl.close(fig)
        else:
            return pl, ax

def plot_dt_map(rxPulses_l, rxList, sourceDepth, ii_select=0, var_mode=True,
               ppp_mode=True,
               log_mode=True, show_mode = True, cl_mode = False, map_cut = [100, 290, None, 170],
               fname_out = None, path2plots = '', vmin=None, vmax=None,
               cmap='viridis', interp_mode='spline36', title_suffix = None,
               figsize =(12, 6),fontsize=16, labelsize=12):
    if var_mode == True:
        time_w = rxPulses_l[:,1] - rxPulses_l[:,0]
    else:
        time_w = rxPulses_l[ii_select, :, 1] - rxPulses_l[ii_select, :, 0]
    x_rx_arr = rxList[:,0]
    x_rx_un = np.unique(x_rx_arr)
    z_rx_arr = rxList[:,1]
    z_rx_un = np.unique(z_rx_arr)
    nBins_x = len(x_rx_un)
    nBins_z = len(z_rx_un)

    hist2d, x_bins, z_bins = np.histogram2d(x_rx_arr, z_rx_arr,
                                            bins=(nBins_x, nBins_z),
                                            weights=time_w)

    hist2d = np.transpose(hist2d)

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
    z_symbol = '$\Delta t_{DR}$'
    title_prefix = 'Relative time offset ' + z_symbol + ' prop. time variance '
    title_pl = title_prefix + str(z_symbol) + ', $z_{tx} = ' + str(sourceDepth) + '\,  \mathrm{m}$'
    z_symbol += ' [ns]'
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

    #ax.set_aspect(1)
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
        if cl_mode == True:
            pl.close(fig)
        else:
            return pl, ax


def plot_dt_map2(rxPulses_l, rxList, sourceDepth, ii_select=0, var_mode=True,
               ppp_mode=True,
               log_mode=True, show_mode = True, cl_mode = False, map_cut = [100, 290, None, 170],
               fname_out = None, path2plots = '', vmin=None, vmax=None,
               cmap='viridis', interp_mode='spline36', title_suffix = None,
               figsize =(12, 6),fontsize=16, labelsize=12):

    time_w2 = rxPulses_l[:,1]
    time_w1 = rxPulses_l[:,0]
    x_rx_arr = rxList[:,0]
    x_rx_un = np.unique(x_rx_arr)
    z_rx_arr = rxList[:,1]
    z_rx_un = np.unique(z_rx_arr)
    nBins_x = len(x_rx_un)
    nBins_z = len(z_rx_un)

    hist2d1, x_bins, z_bins = np.histogram2d(x_rx_arr, z_rx_arr,
                                            bins=(nBins_x, nBins_z),
                                            weights=time_w1)
    hist2d2, x_bins, z_bins = np.histogram2d(x_rx_arr, z_rx_arr,
                                            bins=(nBins_x, nBins_z),
                                            weights=time_w2)
    hist2d = hist2d1 + hist2d2
    hist2d = np.transpose(hist2d)

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
    z_symbol = '$\Delta t_{DR}$'
    title_prefix = 'Relative time offset ' + z_symbol + ' prop. time variance '
    title_pl = title_prefix + str(z_symbol) + ', $z_{tx} = ' + str(sourceDepth) + '\,  \mathrm{m}$'
    z_symbol += ' [ns]'
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

    #ax.set_aspect(1)
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
        if cl_mode == True:
            pl.close(fig)
        else:
            return pl, ax