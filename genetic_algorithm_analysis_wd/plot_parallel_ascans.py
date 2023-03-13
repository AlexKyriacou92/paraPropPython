import numpy as np
import sys
sys.path.append('../')

import paraPropPython as ppp
from receiver import receiver
from data import bscan, bscan_rxList
import os
import scipy.signal as signal
import util
import h5py
import matplotlib.pyplot as pl
import peakutils as pku
import sys
from fitness_function import fitness_pulse_FT_data

if len(sys.argv) == 4:
    fname = sys.argv[1]
    fname_FT = sys.argv[2]
    pathto_plots = sys.argv[3]
    fname_nmatrix = pathto_plots + 'test_nmatrix_data_1.h5'
    ii_gen = 87
    jj_ind = 15
elif len(sys.argv) == 7:
    fname = sys.argv[1]
    fname_FT = sys.argv[2]
    pathto_plots = sys.argv[3]
    fname_nmatrix = sys.argv[4]
    ii_gen = int(sys.argv[5])
    jj_ind = int(sys.argv[6])
else:
    print('error: must enter python plot_parallel_bscan.py fname')
    sys.exit()
vmin0 = -100
vmax0 = 10

fname_h5 = fname
sim_bscan = bscan_rxList()
sim_bscan.load_sim(fname_h5)
tspace = sim_bscan.tx_signal.tspace
tx_depths = sim_bscan.tx_depths
nSamples = sim_bscan.nSamples
nDepths = len(tx_depths)
rxList = sim_bscan.rxList
print(rxList[0].x)

rx_depths = []
for i in range(len(rxList)):
    rx_depths.append(rxList[i].z)

bscan_sig = sim_bscan.bscan_sig

bscan_plot = np.zeros((nDepths, nSamples), dtype='complex')

hdf_FT = h5py.File(fname_FT,'r')
fftArray = np.array(hdf_FT.get('fftArray'))
tx_depths_FT = np.array(hdf_FT.get('txDepths'))
rx_depths_FT = np.array(hdf_FT.get('rxDepths'))
rx_ranges_FT = np.array(hdf_FT.get('rxRanges'))
nDepths_FT = len(tx_depths_FT)
tspace_FT = np.array(hdf_FT.get('tspace'))
bscan_FT = np.zeros((nDepths, nSamples), dtype='complex')
hdf_FT.close()

S_list = []
S_delta_list = []
S_accum = []
delta_t_list  = []
S_corr = 0

p_sim_list = []
p_FFT_list = []

t_sim_list = []
t_FFT_list = []

if os.path.isfile(fname_nmatrix) == True:
    hdf_nmatrix = h5py.File(fname_nmatrix,'r')
    n_matrix = np.array(hdf_nmatrix.get('n_profile_matrix'))
    z_profile = np.array(hdf_nmatrix.get('z_profile'))
    n_profile_best = n_matrix[ii_gen,jj_ind]
for i in range(nDepths):
    print(i, tx_depths[i])
    jj_rx = util.findNearest(rx_depths, tx_depths[i])
    bscan_plot[i] = bscan_sig[i,jj_rx]

    ii_tx_FT = util.findNearest(tx_depths_FT, tx_depths[i])
    ii_rx_FT = util.findNearest(rx_depths_FT, tx_depths[i])
    bscan_FT[i] = fftArray[ii_tx_FT]

    sig_sim = bscan_plot[i]
    sigFFT = bscan_FT[i]

    sig_sim_power = abs(sig_sim)**2
    sigFFT_power = abs(sigFFT)**2

    sim_peak_inds = pku.indexes(sig_sim_power, thres = 0.5)
    FFT_peaks_inds = pku.indexes(abs(sigFFT_power)**2, thres = 0.5)

    t_sim_peaks = tspace[sim_peak_inds]
    p_sim_peaks = sig_sim_power[sim_peak_inds]

    t_FFT_peaks = tspace_FT[FFT_peaks_inds]
    p_FFT_peaks = sigFFT_power[FFT_peaks_inds]

    jj_sim = np.argmax(sig_sim_power)
    jj_FFT = np.argmax(sigFFT_power)
    t_max_sim = tspace[jj_sim]
    t_max_FFT = tspace_FT[jj_FFT]*1e9
    p_max_sim = sig_sim_power[jj_sim]
    p_max_FFT = sigFFT_power[jj_FFT]
    p_sim_list.append(p_max_sim)
    p_FFT_list.append(p_max_FFT)
    t_sim_list.append(t_max_sim)
    t_FFT_list.append(t_max_FFT)

    delta_t = t_max_FFT - t_max_sim
    delta_t_list.append(delta_t)

    print(len(sig_sim), len(sigFFT))
    S_corr_ijk = fitness_pulse_FT_data(sig_sim=sig_sim, sig_data=sigFFT)
    S_list.append(S_corr_ijk)
    S_corr += S_corr_ijk
    S_accum.append(S_corr)

    print('S_ij = ', S_corr_ijk)
    print('S_total = ', S_corr)
    delta_sig = (sigFFT_power - sig_sim_power)**2
    delta_sig_norm = sum(delta_sig)
    S_delta = 1/delta_sig_norm
    S_delta_list.append(S_delta)

    fig = pl.figure(figsize=(8,5),dpi=200)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    '''
    ax1.plot(tspace, 10*np.log10(abs(bscan_plot[i])**2),c='b',label='Sim')
    ax2.plot(tspace_FT*1e9, 10*np.log10(abs(bscan_FT[i])**2),c='r',label='Data')
    '''
    ax1.set_title(r'$Z_{tx} =$ ' + str(tx_depths[i]) + r' m, R = ' + str(rx_ranges_FT[i]) + r' m, $\log_{10}(S) = $ ' + str(round(np.log10(S_corr_ijk),2)))

    ax1.plot(tspace, sig_sim_power,c='b',label='Sim')
    ax2.plot(tspace_FT*1e9, sigFFT_power, c='r',label='Data')

    ax1.grid()
    ax1.legend(loc='upper left')
    ax2.legend()
    fname_plot = pathto_plots + 'ascan-z=' + str(round(tx_depths[i],1)) + 'm.png'
    pl.savefig(fname_plot)
    pl.close(fig)

delta_t_list = np.array(delta_t_list)
S_delta_list = np.array(S_delta_list)

fig = pl.figure(figsize=(8,5),dpi=120)
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax1.plot(tx_depths, np.array(S_list)/S_corr,c='b')
ax2.plot(tx_depths, abs(delta_t_list),c='k')
ax1.set_yscale('log')
ax2.set_ylabel(r'$\Delta t$ [ns]')
ax1.set_ylabel('S')
ax1.set_xlabel('Tx Depths [m]')
ax1.grid()
pl.savefig(pathto_plots + 'S_vs_txdepths.png')
pl.close()

fig = pl.figure(figsize=(8,5),dpi=120)
ax1 = fig.add_subplot(111)
ax1.plot(tx_depths, np.array(p_sim_list),'-o',c='b',label='Sim')
ax1.plot(tx_depths, np.array(p_FFT_list),'-o',c='r',label='FFT')
ax1.set_ylabel(r'$P$')
ax1.set_xlabel('Tx Depths [m]')
ax1.grid()
ax1.legend()
pl.savefig(pathto_plots + 'P_linear_depths.png')
pl.close()

B = 0.2
t_res = 1/B

t_sim_list = np.array(t_sim_list)
t_FFT_list = np.array(t_FFT_list)
R = rx_ranges_FT[i]
c0 = 0.3 # m/ns
t_air = R/c0

rel_error_sim = t_res/t_sim_list
rel_error_FFT = t_res/t_FFT_list

n_sim = t_sim_list/t_air
n_FFT = t_FFT_list/t_air
n_sim_err = rel_error_sim*n_sim
n_FFT_err = rel_error_FFT*n_FFT


fig = pl.figure(figsize=(8,5),dpi=120)
ax1 = fig.add_subplot(111)
ax1.errorbar(tx_depths, t_sim_list, t_res, c='b', label='Sim', fmt='-o',alpha=0.5)
ax1.errorbar(tx_depths, t_FFT_list, t_res, c='r', label='FFT', fmt='-o',alpha=0.5)
ax1.set_ylabel(r'$\Delta t $ [ns]')
ax1.set_xlabel('Tx Depths [m]')
ax1.grid()
ax1.legend()
pl.savefig(pathto_plots + 'time_vs_depths.png')
pl.close()

fig = pl.figure(figsize=(8,5),dpi=120)
ax1 = fig.add_subplot(111)
ax1.errorbar(tx_depths, n_sim, n_sim_err, c='b',label='Sim',fmt='-o',alpha=0.5)
ax1.errorbar(tx_depths, n_FFT, n_FFT_err, c='r',label='FFT',fmt='-o',alpha=0.5)
ax1.plot(z_profile, n_profile_best, c='g',label='n_profile best')
ax1.set_ylabel(r'$ n(z) $')
ax1.set_xlabel(r'$z = Z_{TX}$ Depths [m]')
ax1.grid()
ax1.legend()
pl.savefig(pathto_plots + 'ref_index_profile.png')
pl.close()


fig = pl.figure(figsize=(8,5),dpi=120)
ax1 = fig.add_subplot(111)
ax1.plot(tx_depths, 10*np.log10(p_sim_list),c='b',label='Sim')
ax1.plot(tx_depths, 10*np.log10(p_FFT_list),c='r',label='FFT')
ax1.set_ylabel(r'$P$ [dB]')
ax1.set_xlabel('Tx Depths [m]')
ax1.grid()
ax1.legend()
pl.savefig(pathto_plots + 'P_dB_depths.png')
pl.close()

fig = pl.figure(figsize=(8,5),dpi=120)
ax1 = fig.add_subplot(111)
ax1.plot(tx_depths, 10*(np.log10(p_sim_list)-np.log10(p_FFT_list)),c='b')
ax1.set_ylabel(r'$\Delta P$ [dB]')
ax1.set_xlabel('Tx Depths [m]')
ax1.grid()
pl.savefig(pathto_plots + 'P_delta_depths.png')
pl.close()

fig = pl.figure(figsize=(8,5),dpi=120)
ax1 = fig.add_subplot(111)
ax1.scatter(abs(delta_t_list), np.array(S_list))
ax1.grid()
ax1.set_yscale('log')
ax1.set_xlabel(r'$\Delta t$ [ns]')
ax1.set_ylabel('S')
pl.savefig(pathto_plots + 'delta_t_S_correl.png')
pl.close()