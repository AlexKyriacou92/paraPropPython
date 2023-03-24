import numpy as np
import sys

import scipy.signal

from fitness_function import fitness_pulse_FT_data

sys.path.append('../')

import paraPropPython as ppp
from receiver import receiver
from data import bscan, bscan_rxList

import scipy.signal as signal
import util
from util import findNearest
import h5py
import matplotlib.pyplot as pl
import peakutils as pku
import sys
import os
import configparser
from scipy.interpolate import interp1d


def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    return b_1, b_0


def precision_round(number, digits=3):
    power = "{:e}".format(number).split('e')[1]
    return round(number, -(int(power) - digits))

path2dir = sys.argv[1]
fname_simul_report = path2dir + 'simul_report.txt'
with open(fname_simul_report) as f_report:
    next(f_report)
    next(f_report)
    next(f_report)
    next(f_report)
    cols = f_report.readline().split()
    fname_pseudo = path2dir + cols[0]
    fname_nmatrix = path2dir + cols[1]
    print(cols[0],cols[1])


weighting_bool = True
nmatrix_hdf = h5py.File(fname_nmatrix, 'r')
zspace_simul = np.array(nmatrix_hdf['z_profile'])
n_profile_matrix = np.array(nmatrix_hdf['n_profile_matrix'])
nmatrix_hdf.close()

path2rand = path2dir + 'random_bscans/'
fname_list = path2rand + 'bscan_list.txt'

hdf_pseudo = h5py.File(fname_pseudo, 'r')
pseudo_data = np.transpose(hdf_pseudo['n_profile'])
# print(pseudo_data.shape)
z_profile_pseudo = pseudo_data[:, 0].real
n_profile_pseudo = pseudo_data[:, 1].real
# print(n_profile_pseudo)
hdf_pseudo.close()

bscan_pseudo = bscan_rxList()
bscan_pseudo.load_sim(fname_pseudo)
tx_depths = bscan_pseudo.tx_depths
tspace = bscan_pseudo.tspace
rxList = bscan_pseudo.rxList

nRX = len(rxList)
rx_depths = []
rx_ranges = []
for i in range(nRX):
    rx_depths.append(rxList[i].z)
    rx_ranges.append(rxList[i].x)

bscan_list = []
ii_list = []
jj_list = []
S_signal_list = []
S_nprof_list = []

nMax = 10
kk = 0

with open(fname_list) as fin:
    print(fin.readline())
    for line in fin:
        cols = line.split()
        print(cols)
        fname_rand_bscan = path2rand + cols[4]
        if os.path.isfile(fname_rand_bscan) == True:
            if kk < nMax:
                ii_gen = int(cols[0])
                ii_list.append(ii_gen)

                jj_ind = int(cols[1])
                jj_list.append(jj_ind)

                S_signal = float(cols[2])
                print(ii_gen, S_signal)
                S_signal_list.append(S_signal)

                S_nprof = float(cols[3])
                S_nprof_list.append(S_nprof)

                bscan_rand = bscan_rxList()
                bscan_rand.load_sim(fname_rand_bscan)
                bscan_list.append(bscan_rand)
            kk += 1
ind_rank = np.array(S_signal_list).argsort()
S_signal_list = np.array(S_signal_list)
ii_list = np.array(ii_list)
jj_list = np.array(jj_list)
bscan_rand = np.array(bscan_list)
S_nprof_list = np.array(S_nprof_list)

S_signal_list = S_signal_list[ind_rank]
ii_list = ii_list[ind_rank]
jj_list = jj_list[ind_rank]
bscan_rand = bscan_rand[ind_rank]
S_nprof_list = S_nprof_list[ind_rank]

nOutput = len(bscan_list)
print('nOutput = ', nOutput)
inds = np.array(S_signal_list).argsort()

nTX = len(tx_depths)

tx_depth_plot = np.arange(2.0, 12.0+2.0, 2.0)
R_plot = [25, 42]
pseudodata_weights = np.zeros((nOutput, nTX, nRX))
sim_weights = np.zeros((nOutput, nTX, nRX))
W_pseudodata = 0
W_sim = 0
pseudodata_inv_weights = np.ones((nOutput, nTX, nRX))
sim_inv_weights = np.ones((nOutput, nTX, nRX))

s_ij_matrix = np.zeros(shape=(nOutput, nTX, nRX))
s_w_ij_matrix = np.zeros(shape=(nOutput, nTX, nRX))
delta_s = np.zeros(shape=(nOutput, nTX, nRX))
delta_s_w = np.zeros(shape=(nOutput, nTX, nRX))

fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
ax.plot(S_signal_list)
ax.grid()
ax.set_ylabel('S')
fig.savefig(path2dir + 'S_rand.png')
pl.close(fig)

path2dir2 = path2dir + 'random_bscan2'
if os.path.isdir(path2dir2) == False:
    os.system('mkdir ' + path2dir2)
path2dir2 += '/'
for i in range(nOutput):
    bscan_rand = bscan_list[i]

    ii_gen = ii_list[i]
    jj_ind = jj_list[i]

    S_signal = S_signal_list[i]
    S_nprof = S_nprof_list[i]

    for j in range(nTX):
        for k in range(nRX):
            ascan_pseudo = bscan_pseudo.bscan_sig[j, k]
            ascan_rand = bscan_rand.bscan_sig[j, k]

            sim_weights[i, j, k] = sum(abs(ascan_rand))
            W_sim += sim_weights[i,j, k]
            pseudodata_weights[i, j, k] = sum(abs(ascan_pseudo))
            W_pseudodata += pseudodata_weights[i, j, k]

            s_ij_matrix[i,j,k] = fitness_pulse_FT_data(sig_data=ascan_pseudo, sig_sim=ascan_rand,mode='Difference')
            if i == 0:
                delta_s[i,j,k] = s_ij_matrix[i,j,k]
            else:
                delta_s[i,j,k] = s_ij_matrix[i,j,k]/s_ij_matrix[i-1,j,k]
            print(delta_s[i,j,k])
    for j in range(nTX):
        for k in range(nRX):
            if pseudodata_weights[i, j, k] > 0:
                pseudodata_inv_weights[i, j, k] = W_pseudodata / pseudodata_weights[i, j, k]
            else:
                pseudodata_inv_weights[i, j, k] = 0
            if sim_weights[i, j, k] > 0:
                sim_inv_weights[i, j, k] = W_sim / sim_weights[i, j, k]
            else:
                sim_inv_weights[i, j, k] = 0

    for j in range(nTX):
        for k in range(nRX):
            ascan_pseudo = bscan_pseudo.bscan_sig[j, k]
            ascan_rand = bscan_rand.bscan_sig[j, k]

            ascan_pseudo_w = ascan_pseudo * pseudodata_inv_weights[i,j, k]
            ascan_rand_w = ascan_rand * sim_inv_weights[i, j, k]
            s_w_ij_matrix[i,j,k] = fitness_pulse_FT_data(sig_data=ascan_pseudo_w, sig_sim=ascan_rand_w,mode='Difference')
            if i == 0:
                delta_s_w[i,j,k] = s_w_ij_matrix[i,j,k]
            else:
                delta_s_w[i,j,k] = s_w_ij_matrix[i,j,k] - s_w_ij_matrix[i-1,j,k]

for m in range(len(R_plot)):
    print('make plots for range ', R_plot[m])
    rxList_inds = []
    rx_depth_list = []

    for i in range(nRX):
        rx_i = rxList[i]
        if rx_i.x == R_plot[m]:
            rx_depth_list.append(rx_i.z)
            rxList_inds.append(i)

    nRX_depths = len(rx_depth_list)

    n_profile_list = []

    max_s_delta_pulse_list = []

    s_ij_matrix2 = np.zeros(shape=(nOutput, nTX, nRX_depths))
    s_w_ij_matrix2 = np.zeros(shape=(nOutput, nTX, nRX_depths))
    delta_s2 = np.zeros(shape=(nOutput, nTX, nRX_depths))
    delta_s_w2 = np.zeros(shape=(nOutput, nTX, nRX_depths))
    nSamples = bscan_rand.nSamples
    bscan_pseudo2 = np.zeros((nOutput,nTX, nRX_depths, nSamples))
    bscan_rand2 = np.zeros((nOutput,nTX, nRX_depths, nSamples))
    for i in range(nOutput):


        ii_output = i
        s_ij_list = []
        bscan_rand = bscan_list[i]

        ii_gen = ii_list[i]
        jj_ind = jj_list[i]

        S_signal = S_signal_list[i]#*1e6
        S_nprof = S_nprof_list[i]
        S_label = str(round(S_signal, 2)).zfill(9)
        S_rank = str(ii_output + 1).zfill(3)

        print(ii_gen, jj_ind)
        print('S_sig = ', S_signal, np.log10(S_signal))
        print('S_n = ', S_nprof, np.log10(S_nprof))
        nSamples = bscan_pseudo.nSamples


        n_profile_rand = n_profile_matrix[ii_gen, jj_ind]
        n_profile_list.append(n_profile_rand)
        for j in range(nTX):
            for k in range(nRX_depths):
                kk = rxList_inds[k]
                ascan_pseudo = bscan_pseudo.bscan_sig[j,kk]
                ascan_rand = bscan_rand.bscan_sig[j,kk]

                s_ij_matrix2[i,j,k] = s_ij_matrix[i,j,kk]
                s_w_ij_matrix2[i,j,k] = s_w_ij_matrix[i,j,kk]

                delta_s2[i,j,k] = delta_s[i,j,kk]
                delta_s_w2[i,j,k] = delta_s_w[i,j,kk]
                bscan_pseudo2[i,j,k] = bscan_pseudo.bscan_sig[j,kk]
                bscan_rand2[i,j,k] = bscan_rand.bscan_sig[j,kk]

        inds = np.unravel_index(np.argmax(delta_s_w2[i], axis=None), delta_s_w2[i].shape)
        print(inds)
        ii_tx = inds[0]
        ii_rx = inds[1]

        ascan_rand_max = bscan_rand2[i,ii_tx, ii_rx]
        ascan_pseudo_max = bscan_pseudo2[i,ii_tx, ii_rx]

        if i > 0:
            vmax = np.amax(delta_s[i])
            vmin = np.amin(delta_s[i])

            fig = pl.figure(figsize=(15,15),dpi=100)
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224)

            fig_label = '$\Delta S$ = ' + str(S_signal/S_signal_list[i-1]) + '\n'
            s_ij_max = delta_s2[i, ii_tx, ii_rx]
            fig_label += '$\Delta_{ij}$ = ' + str(s_ij_max) + ', $\log(\Delta s_{ij}) = $' + str(np.log10(s_ij_max)) + '\n'
            s_ij_w_max = delta_s_w2[i, ii_tx, ii_rx]
            fig_label += '$\Delta_{w, ij}$ = ' + str(s_ij_w_max) + ', $\log(\Delta s_{w,ij}) = $' + str(np.log10(s_ij_w_max)) + '\n'
            s_ij_w_ratio = s_ij_w_max / np.sum(delta_s_w2[i])
            fig_label += '$\Delta_{w, ij}/S$ = ' + str(s_ij_w_ratio) + ', $\log(\Delta s_{w,ij}/S) = $' + str(
                np.log10(s_ij_w_ratio))
            fig.suptitle(fig_label)
            ii_gen_last = ii_list[i-1]
            jj_ind_last = jj_list[i-1]
            ax1.plot(n_profile_rand, zspace_simul,c='b',label='Ref Index, gen: ' + str(ii_gen) + ' ind: ' + str(jj_ind))
            ax1.plot(n_profile_list[i-1], zspace_simul,c='r', label='Ref Index, gen: ' + str(ii_gen_last) + ' ind: ' + str(jj_ind_last))
            ax1.plot(n_profile_pseudo, z_profile_pseudo,c='k', label='Pseudo Data')
            ax1.grid()
            ax1.set_ylim(18,0)
            ax1.legend()
            ax1.set_xlabel('Ref Index n')
            ax1.set_ylabel('Depth z[m]')

            ax2.scatter(np.zeros(nTX), tx_depths, c='b',label='TX')
            ax2.scatter(rx_ranges, rx_depths,c='r',label='RX')
            x=[0, R_plot[m]]
            y=[tx_depths[ii_tx], rx_depth_list[ii_rx]]
            ax2.plot(x,y,c='k')
            ax2.grid()
            ax2.set_ylim(18,0)
            ax2.set_xlabel('Range x [m]')
            ax2.set_ylabel('Depth z [m]')


            pmesh3 = ax3.imshow(delta_s2[i], aspect='auto', cmap='coolwarm', vmin=vmin, vmax=vmax,
                                extent=[rx_depth_list[0], rx_depth_list[-1], tx_depths[-1], tx_depth_plot[0]])
            cbar3 = pl.colorbar(pmesh3, ax=ax3)
            cbar3.set_label('Local Score $elta s$')
            ax3.set_xticks(rx_depth_list)
            ax3.set_yticks(tx_depths)
            ax3.set_xlabel('Rx Depth')
            ax3.set_ylabel('TX Depth')

            ax4.plot(tspace, ascan_pseudo_max,c='k', label='Pseudo Data')
            ascan_rand_old = bscan_rand2[i-1, ii_tx, ii_rx]
            ax4.plot(tspace, ascan_rand_old,c='r',alpha=0.6, label='Pulse, gen: ' + str(ii_gen_last) + ' ind: ' + str(jj_ind_last))
            ax4.plot(tspace, ascan_rand_max, c='b',alpha=0.6, label='Pulse, gen: ' + str(ii_gen) + ' ind: ' + str(jj_ind))

            if R_plot[m] == 25:
                ax4.set_xlim(80, 200)
            else:
                ax4.set_xlim(180, 300)
            ax4.grid()
            ax4.legend()
            ax4.set_xlabel('Time t [ns]')
            ax4.set_ylabel('Amplitude [u]')

            fig_label= path2dir2 + 'deltaS_rank=' + S_rank + 'S=' + S_label+ '_R=' + str(R_plot[m]) + '_plot.png'
            fig.savefig(fig_label)
            pl.close(fig_label)



for m in range(len(R_plot)):
    print('make plots for range ', R_plot[m])
    rxList_inds = []
    rx_depth_list = []

    for i in range(nRX):
        rx_i = rxList[i]
        if rx_i.x == R_plot[m]:
            rx_depth_list.append(rx_i.z)
            rxList_inds.append(i)

    nRX_depths = len(rx_depth_list)

    n_profile_list = []

    max_s_delta_pulse_list = []

    s_ij_matrix2 = np.zeros(shape=(nOutput, nTX, nRX_depths))
    s_w_ij_matrix2 = np.zeros(shape=(nOutput, nTX, nRX_depths))
    delta_s2 = np.zeros(shape=(nOutput, nTX, nRX_depths))
    delta_s_w2 = np.zeros(shape=(nOutput, nTX, nRX_depths))
    nSamples = bscan_rand.nSamples
    bscan_pseudo2 = np.zeros((nOutput,nTX, nRX_depths, nSamples))
    bscan_rand2 = np.zeros((nOutput,nTX, nRX_depths, nSamples))

    for i in range(nOutput):


        ii_output = i
        s_ij_list = []
        bscan_rand = bscan_list[i]

        ii_gen = ii_list[i]
        jj_ind = jj_list[i]

        S_signal = S_signal_list[i]#*1e6
        S_nprof = S_nprof_list[i]
        S_label = str(round(S_signal, 2)).zfill(9)
        S_rank = str(ii_output + 1).zfill(3)

        print(ii_gen, jj_ind)
        print('S_sig = ', S_signal, np.log10(S_signal))
        print('S_n = ', S_nprof, np.log10(S_nprof))
        nSamples = bscan_pseudo.nSamples


        n_profile_rand = n_profile_matrix[ii_gen, jj_ind]
        n_profile_list.append(n_profile_rand)

        for j in range(nTX):
            for k in range(nRX_depths):
                kk = rxList_inds[k]
                ascan_pseudo = bscan_pseudo.bscan_sig[j,kk]
                ascan_rand = bscan_rand.bscan_sig[j,kk]

                s_ij_matrix2[i,j,k] = s_ij_matrix[i,j,kk]
                s_w_ij_matrix2[i,j,k] = s_w_ij_matrix[i,j,kk]

                delta_s2[i,j,k] = delta_s[i,j,kk]
                delta_s_w2[i,j,k] = delta_s_w[i,j,kk]
                bscan_pseudo2[i,j,k] = bscan_pseudo.bscan_sig[j,kk]
                bscan_rand2[i,j,k] = bscan_rand.bscan_sig[j,kk]

        inds = np.unravel_index(np.argmax(s_w_ij_matrix2[i], axis=None), s_ij_matrix2[i].shape)
        print(inds)
        ii_tx = inds[0]
        ii_rx = inds[1]

        ascan_rand_max = bscan_rand2[i,ii_tx, ii_rx]
        ascan_pseudo_max = bscan_pseudo2[i,ii_tx, ii_rx]

        vmax = np.amax(s_w_ij_matrix2[i])
        vmin = np.amin(s_w_ij_matrix2[i])

        fig = pl.figure(figsize=(15, 15), dpi=100)
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        fig_label = 'S = ' + S_label + '\n'
        s_ij_max = s_ij_matrix2[i, ii_tx, ii_rx]
        fig_label += '$s_{ij}$ = ' + str(s_ij_max) + ', $\log(s_{ij}) = $' + str(np.log10(s_ij_max)) + '\n'
        s_ij_w_max = s_w_ij_matrix2[i, ii_tx, ii_rx]
        fig_label += '$s_{w, ij}$ = ' + str(s_ij_w_max) + ', $\log(s_{w,ij}) = $' + str(np.log10(s_ij_w_max)) + '\n'
        s_ij_w_ratio = s_ij_w_max/S_signal
        fig_label += '$s_{w, ij}/S$ = ' + str(s_ij_w_ratio) + ', $\log(s_{w,ij}/S) = $' + str(np.log10(s_ij_w_ratio)) + '\n'
        fig_label += '$s_{w,ij}/<s_{w}> = $' + str(s_ij_w_ratio/(S_signal/(float(nTX*nRX))))

        print('S=',S_signal)
        print(np.sum(s_w_ij_matrix2[i]))
        print(np.sum(s_ij_matrix2[i]))
        print('')
        fig.suptitle(fig_label)
        ii_gen_last = ii_list[i - 1]
        jj_ind_last = jj_list[i - 1]
        ax1.plot(n_profile_rand, zspace_simul, c='b', label='Ref Index, gen: ' + str(ii_gen) + ' ind: ' + str(jj_ind))
        ax1.plot(n_profile_pseudo, z_profile_pseudo, c='k', label='Pseudo Data')
        ax1.grid()
        ax1.set_ylim(18, 0)
        ax1.legend()
        ax1.set_xlabel('Ref Index n')
        ax1.set_ylabel('Depth z[m]')

        ax2.scatter(np.zeros(nTX), tx_depths, c='b', label='TX')
        ax2.scatter(rx_ranges, rx_depths, c='r', label='RX')
        x = [0, R_plot[m]]
        y = [tx_depths[ii_tx], rx_depth_list[ii_rx]]
        ax2.plot(x, y, c='k')
        ax2.grid()
        ax2.set_ylim(18, 0)
        ax2.set_xlabel('Range x [m]')
        ax2.set_ylabel('Depth z [m]')
        print(np.log10(s_w_ij_matrix2[i]))
        pmesh3 = ax3.imshow(s_w_ij_matrix2[i], aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax,
                            extent=[rx_depth_list[0], rx_depth_list[-1], tx_depths[-1], tx_depth_plot[0]])
        cbar3 = pl.colorbar(pmesh3, ax=ax3)
        cbar3.set_label('Local Score $s$')
        ax3.set_xticks(rx_depth_list)
        ax3.set_yticks(tx_depths)
        ax3.set_xlabel('Rx Depth')
        ax3.set_ylabel('TX Depth')

        ax4.plot(tspace, ascan_pseudo_max, c='k', label='Pseudo Data')
        ax4.plot(tspace, ascan_rand_max, c='b', alpha=0.6, label='Pulse, gen: ' + str(ii_gen) + ' ind: ' + str(jj_ind))
        ax4.grid()
        ax4.legend()
        ax4.set_xlabel('Time t [ns]')
        ax4.set_ylabel('Amplitude [u]')
        if R_plot[m] == 25:
            ax4.set_xlim(80, 200)
        else:
            ax4.set_xlim(180, 300)
        fig_label = path2dir2 + 'S_rank=' + S_rank + 'S=' + S_label + '_R=' + str(R_plot[m]) + '_plot.png'
        fig.savefig(fig_label)
        pl.close(fig_label)
