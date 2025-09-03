import numpy as np
import math
import h5py
import sys
from sys import argv, exit
import configparser
import os
from matplotlib import pyplot as pl
from scipy.signal import butter, lfilter
from numpy.lib.format import open_memmap
import scipy.signal as sig

def butterBandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butterBandpassFilter(data, lowcut, highcut, fs, order=3):
    b, a = butterBandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def findNearest(x, xi):
    return np.argmin(abs(x-xi))

def create_memmap(file, dimensions, data_type ='complex'):
    A = open_memmap(file, shape = dimensions, mode='w+', dtype = data_type)
    return A

def do_IR0(data, tspace, IR_spec, IR_freq):
    spec0 = np.fft.fft(data)
    nSamples = len(tspace)
    dt = tspace[1] - tspace[0]
    fspace0 = np.fft.fftfreq(nSamples, dt)
    spec_shift = np.fft.fftshift(spec0)
    freq_shift = np.fft.fftshift(fspace0)
    nMid = findNearest(freq_shift, 0)
    freq_shift_pos = freq_shift[nMid:]
    freq_shift_neg = -1 * np.flip(freq_shift[:nMid])

    nFreq_pos = len(freq_shift_pos)
    nFreq_neg = len(freq_shift_neg)
    IR_arr = np.zeros(nSamples)
    IR_pos = np.zeros(nFreq_pos)
    IR_neg = np.zeros(nFreq_neg)
    for i in range(nFreq_pos):
        freq_plus = freq_shift_pos[i]
        #freq_neg = freq_shift_neg[j_neg]
        if freq_plus < min(IR_freq):
            IR_pos[i] = 0
        elif freq_plus > max(IR_freq):
            IR_pos[i] = 0
        elif freq_plus <= max(IR_freq) and freq_plus >= min(IR_freq):
            k_pos = findNearest(IR_freq, freq_plus)
            IR_pos[i] = IR_spec[k_pos]
    for i in range(nFreq_neg):
        freq_neg = freq_shift_neg[i]
        j_neg = findNearest(freq_shift_neg, freq_neg)
        if freq_plus < min(IR_freq):
            IR_neg[i] = 0
        elif freq_plus > max(IR_freq):
            IR_neg[i] = 0
        elif freq_plus <= max(IR_freq) and freq_plus >= min(IR_freq):
            k_neg = findNearest(IR_freq, freq_plus)
            IR_neg[i] = IR_spec[k_neg]
    IR_arr[nMid:] = IR_pos
    IR_arr[:nMid] = np.flip(IR_neg)
    spec_shift *= IR_arr
    spec_out = np.fft.ifftshift(spec_shift)
    data_out = np.fft.ifft(spec_out)
    return data_out, spec_shift, IR_arr

def do_IR(data, tspace, IR_spec, IR_freq):
    spec0 = np.fft.fft(data)
    nSamples = len(tspace)
    dt = tspace[1] - tspace[0]
    fspace0 = np.fft.fftfreq(nSamples, dt)
    spec_shift = np.fft.fftshift(spec0)
    freq_shift = np.fft.fftshift(fspace0)
    nMid = findNearest(freq_shift, 0)
    freq_shift_pos = freq_shift[nMid:]
    freq_shift_neg = -1 * np.flip(freq_shift[:nMid])

    nFreq_pos = len(freq_shift_pos)
    nFreq_neg = len(freq_shift_neg)
    IR_arr = np.zeros(nSamples)
    IR_pos = np.zeros(nFreq_pos)
    IR_neg = np.zeros(nFreq_neg)
    fmin = min(IR_freq)
    fmax = max(IR_freq)
    IR_pos = np.interp(freq_shift_pos, IR_freq, IR_spec)
    IR_neg = np.interp(freq_shift_neg, IR_freq, IR_spec)

    IR_arr[nMid:] = IR_pos
    IR_arr[:nMid] = np.flip(IR_neg)
    spec_shift *= IR_arr
    spec_out = np.fft.ifftshift(spec_shift)
    data_out = np.fft.ifft(spec_out)
    return data_out, spec_shift, IR_arr

def get_pulse_from_file(fname_hdf, x, z, pol_mode='None'):
    with h5py.File(fname_hdf,'r') as hdf_in:
        if pol_mode == 'None':
            rxPulses = np.array(hdf_in['rxPulses'])
        elif pol_mode == 'zpol':
            rxPulses = np.array(hdf_in['rxPulses_z'])
        elif pol_mode == 'rpol':
            rxPulses = np.array(hdf_in['rxPulses_r'])
        else:
            print('error! enter polarizaiton mode: None, zpol or hpol')
            return -1
        rxList = np.array(hdf_in['rxList'])
        tspace0 = np.array(hdf_in['tspace'])
        nRx = len(rxList)

    dr_list = np.zeros(nRx)
    for i in range(nRx):
        x_rx_i = rxList[i, 0]
        z_rx_i = rxList[i, 1]
        dx = abs(x_rx_i - x)
        dz = abs(z_rx_i - z)
        dr_sq = dx ** 2 + dz ** 2
        dr = np.sqrt(dr_sq)
        dr_list[i] = dr
    ii_rx = np.argmin(dr_list)

    x_actual = rxList[ii_rx, 0]
    z_actual = rxList[ii_rx, 1]

    dt0 = tspace0[1] - tspace0[0]
    dt = dt0 * 3.728
    t_max0 = max(tspace0)
    t_max = t_max0 * 3.728
    nSamples = len(tspace0)
    tspace = np.linspace(0, t_max, nSamples)

    if (z_actual - z == 0) and (x_actual - x == 0):
        print('Match, RX found at (', x_actual,',', z_actual, ')')
    else:
        print('Warning!, mismatch between RX input and RX selected')
        print('Input RX: (', x, ',', z, '), Actual RX: (', x_actual, ',', z_actual, ')')
    rxPulse_out = rxPulses[ii_rx]
    return rxPulse_out, tspace

def get_pulse_from_file_2(fname_hdf, x, z, pol_mode='None'):
    with h5py.File(fname_hdf,'r') as hdf_in:
        if pol_mode == 'None':
            rxPulses = np.array(hdf_in['rxPulses'])
        elif pol_mode == 'zpol':
            rxPulses = np.array(hdf_in['rxPulses_z'])
        elif pol_mode == 'rpol':
            rxPulses = np.array(hdf_in['rxPulses_r'])
        else:
            print('error! enter polarizaiton mode: None, zpol or hpol')
            return -1
        rxList = np.array(hdf_in['rxList'])
        if 'tspace_actual' in hdf_in.keys() == True:
            tspace_actual = np.array(hdf_in['tspace_actual'])
        else:
            tspace_actual = np.array(hdf_in['tspace_meep'])
        nRx = len(rxList)

    dr_list = np.zeros(nRx)
    for i in range(nRx):
        x_rx_i = rxList[i, 0]
        z_rx_i = rxList[i, 1]
        dx = abs(x_rx_i - x)
        dz = abs(z_rx_i - z)
        dr_sq = dx ** 2 + dz ** 2
        dr = np.sqrt(dr_sq)
        dr_list[i] = dr
    ii_rx = np.argmin(dr_list)

    x_actual = rxList[ii_rx, 0]
    z_actual = rxList[ii_rx, 1]

    if (z_actual - z == 0) and (x_actual - x == 0):
        print('Match, RX found at (', x_actual,',', z_actual, ')')
    else:
        print('Warning!, mismatch between RX input and RX selected')
        print('Input RX: (', x, ',', z, '), Actual RX: (', x_actual, ',', z_actual, ')')
    rxPulse_out = rxPulses[ii_rx]
    tspace_meep = tspace_actual[ii_rx]
    c_mGHz = 0.3
    tspace = tspace_meep / c_mGHz
    return rxPulse_out, tspace

def get_pulse_from_file_3(fname_hdf, x, z, pol_mode='None'):
    '''
    This is a special function for buggy simulations -> some unknown bug causes time-space sims to only save a scalar
    value for tspace_meep and tspace (tspace_ns) -> however I have dt_ns and nSamples -> that will let me get tspace
    Args:
        fname_hdf:
        x:
        z:
        pol_mode:

    Returns:

    '''
    with h5py.File(fname_hdf,'r') as hdf_in:
        if pol_mode == 'None':
            rxPulses = np.array(hdf_in['rxPulses'])
        elif pol_mode == 'zpol':
            rxPulses = np.array(hdf_in['rxPulses_z'])
        elif pol_mode == 'rpol':
            rxPulses = np.array(hdf_in['rxPulses_r'])
        else:
            print('error! enter polarizaiton mode: None, zpol or hpol')
            return -1
        rxList = np.array(hdf_in['rxList'])
        nSamples = len(rxPulses[0])
        dt = hdf_in.attrs['dt'] # dt is defined in ns
        tspace_actual = np.linspace(0, dt*nSamples, nSamples)
        nRx = len(rxList)

    dr_list = np.zeros(nRx)
    for i in range(nRx):
        x_rx_i = rxList[i, 0]
        z_rx_i = rxList[i, 1]
        dx = abs(x_rx_i - x)
        dz = abs(z_rx_i - z)
        dr_sq = dx ** 2 + dz ** 2
        dr = np.sqrt(dr_sq)
        dr_list[i] = dr
    ii_rx = np.argmin(dr_list)

    x_actual = rxList[ii_rx, 0]
    z_actual = rxList[ii_rx, 1]

    if (z_actual - z == 0) and (x_actual - x == 0):
        print('Match, RX found at (', x_actual,',', z_actual, ')')
    else:
        print('Warning!, mismatch between RX input and RX selected')
        print('Input RX: (', x, ',', z, '), Actual RX: (', x_actual, ',', z_actual, ')')
    rxPulse_out = rxPulses[ii_rx]
    #tspace_meep = tspace_actual[ii_rx]
    #tspace_meep = tspace_actual
    c_mGHz = 0.3
    #tspace = tspace_meep / c_mGHz
    tspace = tspace_actual
    return rxPulse_out, tspace

def get_index_from_file(fname_hdf, x, z):
    with h5py.File(fname_hdf,'r') as hdf_in:
        rxList = np.array(hdf_in['rxList'])
        nRx = len(rxList)

    dr_list = np.zeros(nRx)
    for i in range(nRx):
        x_rx_i = rxList[i, 0]
        z_rx_i = rxList[i, 1]
        dx = abs(x_rx_i - x)
        dz = abs(z_rx_i - z)
        dr_sq = dx ** 2 + dz ** 2
        dr = np.sqrt(dr_sq)
        dr_list[i] = dr
    ii_rx = np.argmin(dr_list)
    return ii_rx

def get_index_from_rxList(rxList, x, z):
    nRx = len(rxList)
    dr_list = np.zeros(nRx)
    for i in range(nRx):
        x_rx_i = rxList[i, 0]
        z_rx_i = rxList[i, 1]
        dx = abs(x_rx_i - x)
        dz = abs(z_rx_i - z)
        dr_sq = dx ** 2 + dz ** 2
        dr = np.sqrt(dr_sq)
        dr_list[i] = dr
    ii_rx = np.argmin(dr_list)
    return ii_rx

def get_pulse_from_mmap(rxPulses, rxList, x, z):
    ii_rx = get_index_from_rxList(rxList,x,z)
    return rxPulses[ii_rx]

def get_true_rx(fname_hdf, x, z):
    with h5py.File(fname_hdf,'r') as hdf_in:
        rxList = np.array(hdf_in['rxList'])
        nRx = len(rxList)

    dr_list = np.zeros(nRx)
    for i in range(nRx):
        x_rx_i = rxList[i, 0]
        z_rx_i = rxList[i, 1]
        dx = abs(x_rx_i - x)
        dz = abs(z_rx_i - z)
        dr_sq = dx ** 2 + dz ** 2
        dr = np.sqrt(dr_sq)
        dr_list[i] = dr
    ii_rx = np.argmin(dr_list)

    x_actual = rxList[ii_rx, 0]
    z_actual = rxList[ii_rx, 1]
    return x_actual, z_actual

def cut_arr(y, x, xmin, xmax):
    imin = findNearest(x, xmin)
    imax = findNearest(x, xmax)
    return y[imin:imax]

def cut_arr2(y, x, xmin, xmax):
    imin = findNearest(x, xmin)
    imax = findNearest(x, xmax)
    return y[imin:imax], x[imin:imax]

def cut_arr_x(x, xmin, xmax):
    imin = findNearest(x, xmin)
    imax = findNearest(x, xmax)
    return x[imin:imax]

def get_tspace_from_file(fname_hdf):
    with h5py.File(fname_hdf, 'r') as hdf_in:
        tspace = np.array(hdf_in['tspace'])
    tspace  *= 3.728
    return tspace

def get_tspace(fname_hdf):
    '''
    Calculates tspace from nSamples & dt
    Args:
        fname_hdf: Input HDF File

    Returns:

    '''
    with h5py.File(fname_hdf, 'r') as hdf_in:
        if 'tspace_meep' in hdf_in:
            tspace_meep = np.array(hdf_in['tspace_meep'])
            nSamples = len(tspace_meep)

        else:
            rxPulses = np.array(hdf_in['rxPulses_z'])
            nSamples = len(rxPulses[0])
        dt = float(hdf_in.attrs['dt'])
    tspace = np.linspace(0, nSamples*dt, nSamples)
    return tspace

def get_rxList_from_file(fname_hdf):
    with h5py.File(fname_hdf, 'r') as hdf_in:
        rxList = np.array(hdf_in['rxList'])
    return rxList

def get_sourceDepth_from_file(fname_hdf):
    with h5py.File(fname_hdf, 'r') as hdf_in:
        sourceDepth = float(hdf_in.attrs['sourceDepth'])
    return sourceDepth

def get_rxPulses_from_file(fname_hdf, pol_mode='zpol'):
    with h5py.File(fname_hdf, 'r') as hdf_in:
        if pol_mode == 'None':
            rxPulses = np.array(hdf_in['rxPulses'])
        elif pol_mode == 'zpol':
            rxPulses = np.array(hdf_in['rxPulses_z'])
        elif pol_mode == 'rpol':
            rxPulses = np.array(hdf_in['rxPulses_r'])
    return rxPulses

def save_rxPulses_mmap(fname_hdf, fname_npy_out, pol_mode='zpol'):
    with h5py.File(fname_hdf, 'r') as hdf_in:
        if pol_mode == 'None':
            rxPulses = np.array(hdf_in['rxPulses'])
        elif pol_mode == 'zpol':
            rxPulses = np.array(hdf_in['rxPulses_z'])
        elif pol_mode == 'rpol':
            rxPulses = np.array(hdf_in['rxPulses_r'])
    nRx = len(rxPulses)
    nSamples = len(rxPulses[0])
    rxPulse_mmap = create_memmap(fname_npy_out, dimensions=(nRx, nSamples), data_type='complex')

    for i in range(nRx):
        rxPulse_mmap[i] = rxPulses[i]
    return rxPulse_mmap

def get_nProfiles_from_file(fname_hdf):
    with h5py.File(fname_hdf, 'r') as hdf_in:
        nProfile = np.array(hdf_in['nProfile'])
        zProfile = np.array(hdf_in['zProfile'])
    return nProfile, zProfile

def get_spec_from_file(fname_hdf, x, z, pol_mode='zpol'):
    with h5py.File(fname_hdf,'r') as hdf_in:
        if pol_mode == 'None':
            rxPulses = np.array(hdf_in['rxPulses'])
        elif pol_mode == 'zpol':
            rxPulses = np.array(hdf_in['rxPulses_z'])
        elif pol_mode == 'rpol':
            rxPulses = np.array(hdf_in['rxPulses_r'])
        else:
            print('error! enter polarizaiton mode: None, zpol or hpol')
            return -1
        rxList = np.array(hdf_in['rxList'])
        tspace = np.array(hdf_in['tspace'])
        nRx = len(rxList)

    dr_list = np.zeros(nRx)
    for i in range(nRx):
        x_rx_i = rxList[i, 0]
        z_rx_i = rxList[i, 1]
        dx = abs(x_rx_i - x)
        dz = abs(z_rx_i - z)
        dr_sq = dx ** 2 + dz ** 2
        dr = np.sqrt(dr_sq)
        dr_list[i] = dr
    ii_rx = np.argmin(dr_list)

    x_actual = rxList[ii_rx, 0]
    z_actual = rxList[ii_rx, 1]

    if (z_actual - z == 0) and (x_actual - x == 0):
        print('Match, RX found at (', x_actual, ',', z_actual, ')')
    else:
        print('Warning!, mismatch between RX input and RX selected')
        print('Input RX: (', x, ',', z, '), Actual RX: (', x_actual, ',', z_actual, ')')
    rxPulse_out = rxPulses[ii_rx]
    nSamples = len(rxPulse_out)
    dt = (tspace[1] - tspace[0]) * 3.728

    rxPulse_out = butterBandpassFilter(rxPulse_out, 0.05, 0.3, fs=1/dt)
    rxSpec_out = np.fft.rfft(rxPulse_out)
    rxSpec_out /= float(nSamples)
    fspace = np.fft.rfftfreq(nSamples, dt)
    return rxSpec_out, fspace

def get_tx_pulse(fname_hdf):
    with h5py.File(fname_hdf, 'r') as input_hdf:
        tx_pulse = np.array(input_hdf['txPulse'])
        tspace = np.array(input_hdf['tspace'])
    return tx_pulse, tspace

def get_tspace_meep_from_file(fname_hdf):
    with h5py.File(fname_hdf, 'r') as input_hdf:
        tspace_meep = np.array(input_hdf['tspace_meep'])
    return tspace_meep

def butterHighpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sig.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butterHighpassFilter(data, cutoff, fs, order=5):
    b, a = butterHighpass(cutoff, fs, order=order)
    y = sig.filtfilt(b, a, data)
    return y

def butterLowpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='low')
    return b, a


def butterLowpassFilter(data, highcut, fs, order=3):
    b, a = butterLowpass(highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def hilbertTransform(V):
    return np.imag(sig.hilbert(V))

def hilbertEnvelope(V):
    pulse_re = V.real
    pulse_im = hilbertTransform(pulse_re)
    pulse_c = pulse_re + 1j*pulse_im
    return abs(pulse_c)

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
    print(tmin, tmax)
    return np.sum(abs(pulse_v_cut)**2)

def get_2peaks(pulse_rx, tspace, t_cut, t_ratio=2, plot_mode=False):
    pulse_rx_abs = abs(pulse_rx)
    ii_1 = np.argmax(pulse_rx_abs)
    #amp_max_D = np.max(pulse_rx_abs)
    t_max_1 = tspace[ii_1]
    t_end = tspace[-1]

    pulse_rx_roll = shift_t(pulse_rx, tspace, -t_max_1)
    pulse_rx_abs_roll = shift_t(pulse_rx_abs, tspace, -t_max_1)

    t_cut_R = t_ratio*t_cut
    pulse_rx_abs_cut, tspace_cut = cut_arr2(pulse_rx_abs_roll, tspace, t_cut_R, t_end-t_cut_R)
    ii_max_2 = np.argmax(pulse_rx_abs_cut)
    t_max_roll_2 = tspace_cut[ii_max_2] # + t_max_D
    if plot_mode == True:
        fig = pl.figure(figsize=(8,5),dpi=120)
        ax = fig.add_subplot(111)
        ax.plot(tspace, pulse_rx_abs)
        ax.plot(tspace, pulse_rx_roll)
        ax.plot(tspace_cut, pulse_rx_abs_cut)
        pl.show()

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

    '''
    try:
        fluence_D = fluence(pulse_rx, tspace, t_max_D-t_cut, t_max_D+t_cut)
    except:
        print(t_max_1, t_max_roll_2, t_max_D, t_max_R)
        fig = pl.figure(figsize=(8,5),dpi=120)
        ax = fig.add_subplot(111)
        ax.plot(tspace, pulse_rx)
        ax.plot(tspace, pulse_rx_roll)

        ax.axvline(t_max_D,color='k',label='D')
        ax.axvline(t_max_R,color='r',label='R')
        ax.axvline(t_max_1,color='b',label='1',linestyle='--')
        ax.axvline(t_max_1,color='purple',label='2',linestyle='--')

        pl.show()
    '''
    fluence_D = fluence(pulse_rx, tspace, t_max_D - t_cut, t_max_D + t_cut)

    fluence_R = fluence(pulse_rx_abs, tspace, t_max_R-t_cut, t_max_R+t_cut)
    dt_DR = t_max_R - t_max_R
    amp_D = pulse_rx_abs[ii_max_D]
    amp_R = pulse_rx_abs[ii_max_R]
    return fluence_D, fluence_R, amp_D, amp_R, t_max_D, t_max_R, dt_DR

def get_2peaks2(pulse_rx, tspace, t_cut, t_ratio=2):
    pulse_rx_abs = abs(pulse_rx)
    ii_1 = np.argmax(pulse_rx_abs)
    #amp_max_D = np.max(pulse_rx_abs)
    t_max_1 = tspace[ii_1]
    t_end = tspace[-1]

    pulse_rx_roll = shift_t(pulse_rx, tspace, -t_max_1)
    pulse_rx_abs_roll = shift_t(pulse_rx_abs, tspace, -t_max_1)

    t_cut_R = t_ratio*t_cut
    pulse_rx_abs_cut, tspace_cut = cut_arr2(pulse_rx_abs_roll, tspace, t_cut_R, t_end-t_cut_R)
    ii_max_2 = np.argmax(pulse_rx_abs_cut)
    t_max_roll_2 = tspace_cut[ii_max_2] # + t_max_D
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
    fluence_D = fluence(pulse_rx, tspace, t_max_D - t_cut, t_max_D + t_cut)
    fluence_R = fluence(pulse_rx_abs, tspace, t_max_R-t_cut, t_max_R+t_cut)
    fluence_tot = fluence(pulse_rx_abs, tspace)
    dt_DR = t_max_R - t_max_R
    amp_D = pulse_rx_abs[ii_max_D]
    amp_R = pulse_rx_abs[ii_max_R]
    output_l = [fluence_tot, fluence_D, fluence_R, amp_D, amp_R, t_max_D, t_max_R, dt_DR]
    return output_l
def get_2peaks_output(output_l):
    fluence_tot, fluence_D, fluence_R = output_l[0], output_l[1], output_l[2]
    amp_D, amp_R = output_l[3], output_l[4]
    t_max_D, t_max_R, dt_DR = output_l[5], output_l[6], output_l[7]
    return fluence_tot, fluence_D, fluence_R, amp_D, amp_R, t_max_D, t_max_R, dt_DR