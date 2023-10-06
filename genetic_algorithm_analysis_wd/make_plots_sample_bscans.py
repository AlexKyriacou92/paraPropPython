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

nOutput = len(bscan_list)
print('nOutput = ', nOutput)
inds = np.array(S_signal_list).argsort()

nTX = len(tx_depths)
#nOutput = 1
#nOutput = 2
#R_plot = [25]
R_plot = [25,42]
tx_depth_plot = np.arange(2.0, 12.0+2.0, 2.0)
rx_depth_plot = [2, 7, 12, 4, 9, 14]
rx_depths = np.array([[2,7,12], [4,9,14]])
nRX_depths = len(rx_depths[0])

nRanges_plot = len(R_plot)
nDepths_RX_plot = len(rx_depth_plot)
nDepths_TX_plot = len(tx_depth_plot)
s_matrix = np.zeros((nRanges_plot, nDepths_TX_plot, nDepths_RX_plot, nOutput))

ind_rank = np.array(S_signal_list).argsort()
#ind_rank = np.flip(ind_rank)
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

fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
ax.plot(S_signal_list)
ax.grid()
ax.set_ylabel('S')
fig.savefig(path2dir + 'S_rand.png')
pl.close(fig)
sij_matrix_all = np.zeros((nOutput, nRanges_plot, nDepths_TX_plot, nRX_depths))
ascan_rand_delta_list = []
ascan_psuedo_delta_list = []
for i in range(nOutput):
    ii_output = i
    s_ij_list = []
    bscan_rand = bscan_list[i]

    ii_gen = ii_list[i]
    jj_ind = jj_list[i]

    S_signal = S_signal_list[i]#*1e6
    S_nprof = S_nprof_list[i]

    print(ii_gen, jj_ind)
    print('S_sig = ', S_signal, np.log10(S_signal))
    print('S_n = ', S_nprof, np.log10(S_nprof))
    # nTX = 1
    ascan_rand_list = []
    ascan_pseudo_list = []
    coord_list = []

    # Calculate Weights
    pseudodata_inv_weights = np.ones((nTX, nRX))
    sim_inv_weights = np.ones((nTX, nRX))
    if weighting_bool == True:
        pseudodata_weights = np.zeros((nTX, nRX))
        sim_weights = np.zeros((nTX, nRX))
        W_pseudodata = 0
        W_sim = 0

        for j in range(nTX):
            for k in range(nRX):
                ascan_pseudo = bscan_pseudo.bscan_sig[j, k]
                ascan_rand = bscan_rand.bscan_sig[j, k]

                sim_weights[j, k] = sum(abs(ascan_rand))
                W_sim += sim_weights[j, k]
                # sig_pseudodata = sum(abs(bscan_pseudo_data[i,j]))
                pseudodata_weights[j, k] = sum(abs(ascan_pseudo))
                W_pseudodata += pseudodata_weights[j, k]
        pseudodata_inv_weights = W_pseudodata / pseudodata_weights
        sim_inv_weights = W_sim / sim_weights

    sij_matrix = np.zeros((nRanges_plot, nDepths_TX_plot, nRX_depths))
    delta_sij_matrix = np.zeros((nRanges_plot, nDepths_TX_plot, nRX_depths))

    rx_label_matrix = np.zeros((nOutput, nRanges_plot, nDepths_TX_plot, nRX_depths))
    n_counter = 0
    for j in range(nTX):
        for k in range(nRX):
            tx_z = tx_depths[j]
            rx_x = rxList[k].x
            rx_z = rxList[k].z
            coord_list.append([tx_z, rx_x, rx_z])

            ascan_pseudo = bscan_pseudo.bscan_sig[j, k]
            ascan_rand = bscan_rand.bscan_sig[j, k]

            ascan_pseudo_w = ascan_pseudo * pseudodata_inv_weights[j, k]
            ascan_rand_w = ascan_rand * sim_inv_weights[j, k]
            ascan_pseudo_list.append(ascan_pseudo)
            ascan_rand_list.append(ascan_rand)

            s_ij = fitness_pulse_FT_data(sig_sim=ascan_rand_w, sig_data=ascan_pseudo_w, mode='Difference')

            if abs(s_ij) != np.inf and abs(s_ij) != np.nan:
                s_ij_list.append(s_ij)

            ii_tx_depths = util.findNearest(tx_depth_plot, tx_z)
            ii_rx_ranges = util.findNearest(R_plot, rx_x)
            ii_rx_depths = util.findNearest(rx_depths[ii_rx_ranges], rx_z)

            sij_matrix_all[ii_output, ii_rx_ranges, ii_tx_depths, ii_rx_depths] = s_ij
            sij_matrix[ii_rx_ranges, ii_tx_depths, ii_rx_depths] = s_ij

            rx_label_matrix[ii_output, ii_rx_ranges, ii_tx_depths, ii_rx_depths] = n_counter #TODO: Select for pulse
            if ii_output > 1:
                sij_before = sij_matrix_all[ii_output-1, ii_rx_ranges, ii_tx_depths, ii_rx_depths]
                sij_now =  sij_matrix_all[ii_output, ii_rx_ranges, ii_tx_depths, ii_rx_depths]
                sij_ratio = sij_now/sij_before
                delta_sij_matrix[ii_rx_ranges, ii_tx_depths, ii_rx_depths] = sij_ratio
                print(sij_before, sij_now, sij_ratio)

            n_counter += 1

    if abs(S_signal) != np.inf and abs(S_signal) != 0:
        S_label = str(round(S_signal, 2)).zfill(9)
        S_rank = str(ii_output + 1).zfill(3)
        '''
        for m in range(nRanges_plot):
            sij_matrix_m = sij_matrix[m]
            delta_m = delta_sij_matrix[m]


            kk = np.argmax(delta_m)
            print(kk)
            print(len(delta_m))
            print(delta_m[kk])
            ll = np.argmax(delta_m[kk])

            m_rx = rx_label_matrix[ii_output, m, kk, ll]


            print(ll)
            fig = pl.figure(figsize=(15,15),dpi=120)
            fig.suptitle('S = ' + str(S_signal) + ', R = ' + str(R_plot[m]) + ' m')
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            pmesh1 = ax1.imshow(np.log10(sij_matrix_m),aspect='auto', cmap='plasma', extent=[rx_depths[m,0], rx_depths[m,-1], tx_depths[-1], tx_depth_plot[0]])
            pmesh2 = ax2.imshow(np.log10(delta_m),aspect='auto', cmap='coolwarm', extent=[rx_depths[m,0], rx_depths[m,-1], tx_depths[-1], tx_depth_plot[0]])
            cbar1 = pl.colorbar(pmesh1, ax=ax1)
            cbar2 = pl.colorbar(pmesh2, ax=ax2)
            cbar1.set_label('Local Score $\log_{10}(s)$')
            cbar2.set_label('Local Change of Score $\log_{10}(\Delta s)$')
            rx_depths_m = rx_depths[m]
            ax1.set_xticks(rx_depths_m)
            ax2.set_xticks(rx_depths_m)
            ax1.set_yticks(tx_depths)
            ax2.set_yticks(tx_depths)
            ax1.set_xlabel('Rx Depth')
            ax1.set_ylabel('TX Depth')
            ax2.set_xlabel('Rx Depth')
            ax2.set_ylabel('TX Depth')

            x = [1.2, 1.8]
            y = [tx_depths[kk], rx_depths_m[ll]]

            ax3 = fig.add_subplot(223)
            ax3.plot(n_profile_pseudo, z_profile_pseudo, c='k')
            ax3.plot(n_profile_matrix[ii_gen, jj_ind], zspace_simul, c='b')
            ax3.plot(x, y, c='g')

            ax3.grid()
            ax3.set_xlabel('Ref Index')
            ax3.set_ylabel('Depth z [m]')
            ax3.set_ylim(16, 0)
            ax3.set_xlim(1.2, 1.8)
            ax4 = fig.add_subplot(224)
            ax4.plot(tspace, ascan_pseudo_list[m_rx].real)
            ax4.plot(tspace, ascan_rand_list[m_rx].real)
            if ii_output > 0:
                ax4.plot(tspace, ascan_pseudo_list[m_rx].real)
                ax4.plot(tspace, ascan_rand_list[m_rx].real)
            fname_plot_m = path2dir + 's_matrix_rank=' + S_rank + '_S=' + S_label + '_R=' + str(
                    int(R_plot[m])) + '_log.png'

            fig.savefig(fname_plot_m)
            pl.close(fig)
            
            fig = pl.figure(figsize=(15, 15), dpi=120)
            fig.suptitle('S = ' + str(S_signal) + ', R = ' + str(R_plot[m]) + ' m')
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            pmesh1 = ax1.imshow(sij_matrix_m, aspect='auto', cmap='plasma',
                                extent=[rx_depths[m, 0], rx_depths[m, -1], tx_depths[-1], tx_depth_plot[0]])
            pmesh2 = ax2.imshow(delta_m, aspect='auto', cmap='coolwarm',
                                extent=[rx_depths[m, 0], rx_depths[m, -1], tx_depths[-1], tx_depth_plot[0]])
            cbar1 = pl.colorbar(pmesh1, ax=ax1)
            cbar2 = pl.colorbar(pmesh2, ax=ax2)
            # cbar1 = pl.colorbar(pmesh1)
            # cbar2 = pl.colorbar(pmesh2)
            cbar1.set_label('Local Score $s$')
            cbar2.set_label('Local Change of Score $\Delta s$')
            rx_depths_m = rx_depths[m]
            ax1.set_xticks(rx_depths_m)
            ax2.set_xticks(rx_depths_m)
            ax1.set_yticks(tx_depths)
            ax2.set_yticks(tx_depths)
            ax1.set_xlabel('Rx Depth')
            ax1.set_ylabel('TX Depth')
            ax2.set_xlabel('Rx Depth')
            ax2.set_ylabel('TX Depth')

            ax3 = fig.add_subplot(223)
            ax3.plot(n_profile_pseudo, z_profile_pseudo, c='k')
            ax3.plot(n_profile_matrix[ii_gen, jj_ind], zspace_simul, c='b')
            ax3.grid()
            ax3.set_xlabel('Ref Index')
            ax3.set_ylabel('Depth z [m]')
            ax3.plot(x, y, c='g')
            ax3.set_xlim(1.2, 1.8)
            ax3.set_ylim(16, 0)

            ax4 = fig.add_subplot(224)

            if ii_output == 0:
                ax4.plot(n_profile_pseudo, z_profile_pseudo, c='k')
                ax4.plot(n_profile_matrix[ii_gen, jj_ind], zspace_simul, c='b')
                ax4.grid()
                ax4.plot(x,y,c='g')

                ax4.set_xlabel('Ref Index')
                ax4.set_ylabel('Depth z [m]')
                ax4.set_ylim(16, 0)
                ax4.set_xlim(1.2, 1.8)

            else:
                ax4.plot(n_profile_pseudo, z_profile_pseudo, c='k')
                ax4.plot(n_profile_matrix[ii_gen, jj_ind], zspace_simul, c='b')
                ax4.plot(n_profile_matrix[ii_list[ii_output - 1], jj_list[ii_output-1]], zspace_simul, c='r')
                ax4.plot(x,y,c='g')

                ax4.grid()
                ax4.set_xlabel('Ref Index')
                ax4.set_ylabel('Depth z [m]')
                ax4.set_ylim(16, 0)
                ax4.set_xlim(1.2, 1.8)

            fname_plot_m = path2dir + 's_matrix_rank=' + S_rank + '_S=' + S_label + '_R=' + str(
                int(R_plot[m])) + '_log.png'

            fig.savefig(fname_plot_m)
            pl.close(fig)

            fname_plot_m = path2dir + 's_matrix_rank=' + S_rank + '_S=' + S_label + '_R=' + str(
                int(R_plot[m])) + '_lin.png'
            fig.savefig(fname_plot_m)
            pl.close(fig)
            
        '''
        s_ij_list = np.array(s_ij_list)
        # S_signal = sum(s_ij_list)


        print('Percentage of score:', sum(s_ij_list) / S_signal * 100, '%')
        print('Percentage in Maximum:', max(s_ij_list) / S_signal * 100, '%')
        s_ij_list /= S_signal
        ii_top = np.argmax(s_ij_list)
        ii_bottom = np.argmin(s_ij_list)
        ii_mid = np.argmin(abs(s_ij_list - np.median(s_ij_list)))

        fig = pl.figure(figsize=(10, 5), dpi=100)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.hist(s_ij_list, bins=50, range=(0, 1))
        ax1.grid()
        ax1.set_xlabel('s_{ij}/S')
        ax1.set_yscale('log')

        ax2.hist(np.log10(s_ij_list), bins=50)
        ax2.grid()
        ax2.set_xlabel('s_{ij}/S')
        fig.savefig(path2dir + 'hist-rank:' + S_rank + 'S=' + S_label+'-gen=' + str(ii_gen) + '-ind=' + str(jj_ind) + '.png')
        pl.close(fig)

        # Check Inequality
        fig = pl.figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)
        inds_2 = np.array(s_ij_list).argsort()
        s_ij_list2 = np.flip(s_ij_list[inds_2])
        s_ij_cumul = np.zeros(len(s_ij_list2))
        S_ij_c = 0
        for i in range(len(s_ij_list2)):
            S_ij_c += s_ij_list2[i]
            s_ij_cumul[i] = S_ij_c
        rank = np.arange(0, len(s_ij_list2), 1)
        ax.plot(rank, s_ij_cumul)
        # ax.set_yscale('log')
        ax.set_ylabel('Accumulative Score $S_{n} = \sum_{k}^{n} s_{k}$')
        ax.set_xlabel('Ranking n')
        ax.grid()
        fig.savefig(path2dir + 'accumul-score-' + S_rank + 'S=' + S_label+ '-gen=' + str(ii_gen) + '-ind=' + str(jj_ind) + '.png')

        pl.close(fig)

        tmax = 300
        tmin = 50
        alpha_level = 0.5

        fig = pl.figure(figsize=(12, 8), dpi=120)
        ax1 = fig.add_subplot(131)
        ax1.plot(n_profile_pseudo, z_profile_pseudo, c='k')
        ax1.plot(n_profile_matrix[ii_gen, jj_ind], zspace_simul, c='b')
        ax1.set_ylim(16, 0)
        ax1.set_xlabel('Ref Index')
        ax1.set_ylabel('Depth z [m]')
        ax1.grid()

        ax2 = fig.add_subplot(132)
        print(len(n_profile_matrix[ii_gen, jj_ind]), len(n_profile_pseudo))

        f_interp = interp1d(z_profile_pseudo, n_profile_pseudo)
        n_profile_pseudo2 = f_interp(zspace_simul)
        n_residuals = n_profile_pseudo2-n_profile_matrix[ii_gen, jj_ind]
        ax2.plot(n_residuals, zspace_simul, c='k')
        ax2.axvspan(-0.05, 0.05, alpha=0.5, color='red')

        ax2.set_xlabel('Ref Index Residuals')
        ax2.set_ylabel('Depth z [m]')
        ax2.set_ylim(16,0)
        ax2.grid()

        ax3 = fig.add_subplot(133)
        x_plot_max = [0, coord_list[ii_top][1]]
        z_plot_max = [coord_list[ii_top][0], coord_list[ii_top][2]]
        ax3.plot(x_plot_max, z_plot_max, c='b')

        x_plot_mid = [0, coord_list[ii_mid][1]]
        z_plot_mid = [coord_list[ii_mid][0], coord_list[ii_mid][2]]
        ax3.plot(x_plot_mid, z_plot_mid, c='g')

        x_plot_bottom = [0, coord_list[ii_bottom][1]]
        z_plot_bottom = [coord_list[ii_bottom][0], coord_list[ii_bottom][2]]
        ax3.plot(x_plot_bottom, z_plot_bottom, c='r')
        ax3.scatter(np.zeros(len(tx_depths)), tx_depths, c='k')
        ax3.scatter(rx_ranges, rx_depths, c='k')
        ax3.set_ylim(16, -2)
        ax3.axhline(0, c='k')
        ax3.set_xlabel('Range x [m]')
        ax3.set_ylabel('Depth z [m]')
        ax3.grid()
        fig.savefig(
            path2dir + 'profile-compare-rank' + S_rank + 'S=' + S_label + '-gen=' + str(ii_gen) + '-ind=' + str(
                jj_ind) + '.png')
        pl.close(fig)
        '''
        cross_n = scipy.signal.correlate(n_profile_pseudo2, n_profile_matrix[ii_gen, jj_ind])/float(len(zspace_simul))
        lag = np.linspace(-max(zspace_simul)/2, max(zspace_simul)/2, len(cross_n))
        fig = pl.figure(figsize=(8,5),dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(lag,cross_n,c='k')
        ax.set_title('S=' + str(round(S_signal, 2)).zfill(8) + '-gen=' + str(ii_gen) + '-ind=' + str(
                jj_ind))
        ax.grid()
        fig.savefig(path2dir + 'cross-correl-S=' + str(round(S_signal, 6)).zfill(8) + '-gen=' + str(ii_gen) + '-ind=' + str(
                jj_ind) + '.png')
        pl.close(fig)
        '''
        fig = pl.figure(figsize=(18, 8), dpi=120)

        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        z_tx = coord_list[ii_top][0]
        x_rx = coord_list[ii_top][1]
        z_rx = coord_list[ii_top][2]
        ax1_list = 'Best Result \n'
        ax1_list += '$z_{tx} = $' + str(precision_round(z_tx)) + ' m $x_{rx} = $' + str(
            precision_round(x_rx)) + ' m $z_{rx} = $' + str(precision_round(z_rx)) + ' m\n'
        ax1_list += '$s_{ij}/S$ = ' + str(precision_round(s_ij_list[ii_top])) + '\n$\log_{10}(s_{ij}) = $' + str(
            precision_round(np.log10(s_ij_list[ii_top])))
        ax1.set_title(ax1_list)
        ax1.plot(tspace, ascan_pseudo_list[ii_top].real, c='k')
        ax1.plot(tspace, ascan_rand_list[ii_top].real, c='b', alpha=alpha_level)
        ax1.set_xlabel('Time t [ns]')
        ax1.set_ylabel('Amplitude [u]')
        ax1.grid()
        ax1.set_xlim(tmin, tmax)

        z_tx = coord_list[ii_bottom][0]
        x_rx = coord_list[ii_bottom][1]
        z_rx = coord_list[ii_bottom][2]
        ax2_list = 'Worst Result \n'
        ax2_list += '$z_{tx} = $' + str(precision_round(z_tx)) + ' m $x_{rx} = $' + str(
            precision_round(x_rx)) + ' m $z_{rx} = $' + str(precision_round(z_rx)) + ' m\n'
        ax2_list += '$s_{ij}/S$ = ' + str(precision_round(s_ij_list[ii_bottom])) + '\n$\log_{10}(s_{ij}) = $' + str(
            precision_round(np.log10(s_ij_list[ii_bottom])))
        ax2.set_title(ax2_list)
        ax2.plot(tspace, ascan_pseudo_list[ii_bottom].real, c='k')
        ax2.plot(tspace, ascan_rand_list[ii_bottom].real, c='r', alpha=alpha_level)

        ax2.set_xlabel('Time t [ns]')
        ax2.grid()
        ax2.set_xlim(tmin, tmax)

        z_tx = coord_list[ii_mid][0]
        x_rx = coord_list[ii_mid][1]
        z_rx = coord_list[ii_mid][2]
        ax3_list = 'Median Result \n'
        ax3_list += '$z_{tx} = $' + str(precision_round(z_tx)) + ' m $x_{rx} = $' + str(
            precision_round(x_rx)) + ' m $z_{rx} = $' + str(precision_round(z_rx)) + ' m\n'
        ax3_list += '$s_{ij}/S$ = ' + str(precision_round(s_ij_list[ii_mid])) + '\n$\log_{10}(s_{ij}) = $' + str(
            precision_round(
                np.log10(s_ij_list[ii_mid])))
        ax3.set_title(ax3_list)
        ax3.plot(tspace, ascan_pseudo_list[ii_mid].real, c='k')
        ax3.plot(tspace, ascan_rand_list[ii_mid].real, c='g', alpha=alpha_level)

        ax3.set_xlabel('Time t [ns]')
        ax3.grid()
        ax3.set_xlim(tmin, tmax)
        fig.savefig(
            path2dir + 'signal-compare-rank'+ S_rank + 'S=' + S_label+ '-gen=' + str(ii_gen) + '-ind=' + str(
                jj_ind) + '.png')
        #pl.show()
        pl.close(fig)

        path2plots = path2rand + '/plots'
        if os.path.isdir(path2plots) == False:
            os.system('mkdir ' + path2plots)
        print('')
        for l in range(len(R_plot)):
            for m in range(len(tx_depth_plot)):
                for n in range(len(rx_depth_plot)):
                    ii_tx = util.findNearest(tx_depths, tx_depth_plot[m])
                    ii_rx = 0
                    dR = R_plot[l]
                    dz_rx = tx_depth_plot[m] - rx_depth_plot[n]
                    for a in range(nRX):
                        if rxList[a].x == R_plot[l]:
                            if rx_depth_plot[n] == rxList[a].z:
                                ii_rx = a

                    rx_lmn = rxList[ii_rx]
                    z_tx_actual = tx_depths[ii_tx]
                    z_rx_actual = rx_lmn.z
                    x_rx_actual = rx_lmn.x
                    print('z_tx=', z_tx_actual, 'm, x_rx= ', x_rx_actual, 'm, z_rx = ', z_rx_actual, 'm, S = ',
                          S_signal, round(float(ii_output) / float(nOutput) * 100, 2), '%')

                    path2plots_sub = path2plots + '/' + 'tx=' + str(z_tx_actual) + '_rx_x=' + str(
                        x_rx_actual) + '_rx_z=' + str(z_rx_actual)

                    if os.path.isdir(path2plots_sub) == False:
                        os.system('mkdir ' + path2plots_sub)

                    ascan_pseudo = bscan_pseudo.bscan_sig[ii_tx, ii_rx]
                    ascan_rand = bscan_rand.bscan_sig[ii_tx, ii_rx]

                    ascan_rand_w = ascan_rand * sim_inv_weights[ii_tx, ii_rx]
                    ascan_pseudo_w = ascan_pseudo * pseudodata_inv_weights[ii_tx, ii_rx]
                    s_lmn = fitness_pulse_FT_data(sig_sim=ascan_rand_w, sig_data=ascan_pseudo_w, mode='Difference')
                    print(s_matrix.shape, ii_output)

                    s_matrix[l, m, n, ii_output] = s_lmn

                    fname_plot = path2plots_sub + '/pulse' + S_rank + 'S=' + S_label + '.png'
                    fig_label = '$z_{tx} = $' + str(precision_round(z_tx_actual)) + ' m $x_{rx} = $' + str(
                        precision_round(x_rx_actual)) + ' m $z_{rx} = $' + str(precision_round(z_rx_actual)) + ' m\n'
                    fig_label += '$s_{ij}/S$ = ' + str(precision_round(s_lmn)) + '\n$\log_{10}(s_{ij}) = $' + str(
                        precision_round(np.log10(s_lmn)))

                    fig = pl.figure(figsize=(10, 9), dpi=200)
                    fig.suptitle('S = ' + str(precision_round(round(S_signal, 2))).zfill(8) + ' log(S) = ' + str(
                        precision_round(round(np.log10(S_signal), 2))).zfill(8))
                    ax = fig.add_subplot(221)
                    ax.set_title(fig_label)
                    ax.plot(tspace, ascan_pseudo.real, c='r', label='Pseudo Data')
                    ax.plot(tspace, ascan_rand.real, c='b', alpha=alpha_level, label='Simulation')
                    ax.set_xlabel('Time t [ns]')
                    ax.set_ylabel('Amplitude [u]')
                    ax.grid()
                    ax.legend()
                    ax.set_xlim(tmin, tmax)

                    x_plot_max = [0, x_rx_actual]
                    z_plot_max = [z_tx_actual, z_rx_actual]

                    ax1 = fig.add_subplot(223)
                    ax1.plot(n_profile_pseudo, z_profile_pseudo, c='k', label='Pseudo Data')
                    ax1.plot(n_profile_matrix[ii_gen, jj_ind], zspace_simul, c='b', label='Simulation')
                    x_plot2 = [1.1, 1.9]
                    z_plot2 = z_plot_max
                    ax1.plot(x_plot2, z_plot2, label='direct path (vacuum)', c='g')
                    ax1.set_ylim(16, 0)
                    ax1.set_ylabel('Depth z [m]')
                    ax1.set_xlabel('Ref Index n')
                    ax1.legend()

                    ax1.grid()

                    ax2 = fig.add_subplot(222)

                    ax2.plot(x_plot_max, z_plot_max, label='direct path (vacuum)', c='g')
                    ax2.scatter(np.zeros(len(tx_depths)), tx_depths, c='b', label='TX')
                    ax2.scatter(rx_ranges, rx_depths, c='r', label='RX')
                    ax2.legend()
                    ax2.set_ylabel('Depth z [m]')
                    ax2.set_xlabel('Range x [m]')
                    ax2.set_ylim(16, -2)
                    ax1.set_xlim(1.1, 1.9)
                    ax2.axhline(0, c='k')
                    ax2.grid()

                    ax3 = fig.add_subplot(224)
                    print(len(n_profile_matrix[ii_gen, jj_ind]), len(n_profile_pseudo))
                    f_interp = interp1d(z_profile_pseudo, n_profile_pseudo)
                    n_profile_pseudo2 = f_interp(zspace_simul)
                    n_residuals = n_profile_matrix[ii_gen, jj_ind] - n_profile_pseudo2
                    ax3.plot(n_residuals * 100, zspace_simul, c='b')
                    ax3.axvspan(-5, 5, alpha=0.5, color='red')

                    ax3.set_ylim(16, 0)

                    ax3.set_xlabel('n residuals [%]')
                    ax3.grid()
                    fig.savefig(fname_plot)
                    pl.close(fig)

                    if ii_output + 1 == nOutput:
                        s_list = s_matrix[l, m, n, :]

                        print(len(ii_list), len(s_list))
                        fig = pl.figure(figsize=(10, 10), dpi=120)
                        ax1 = fig.add_subplot(221)
                        ax2 = fig.add_subplot(222)
                        ax3 = fig.add_subplot(223)
                        ax4 = fig.add_subplot(224)

                        r_lin = np.corrcoef(S_signal_list[:nOutput], s_list)[0, 1]
                        r_log = np.corrcoef(np.log10(S_signal_list[:nOutput]), np.log10(s_list))[0, 1]
                        print('Correlation of linear scores', r_lin)
                        print('Correlation of log scores', r_log)

                        ax1.scatter(ii_list[:nOutput], s_list, c='b', label='s_lmn')
                        ax2.scatter(ii_list[:nOutput], S_signal_list[:nOutput], c='r', label='S')
                        ax1.grid()
                        ax2.grid()
                        ax1.set_ylabel('TX-RX Local Fitness Score $s_{tx,rx}$')
                        ax2.set_ylabel('Global Fitness Score $S$')
                        ax1.set_xlabel('Generation')
                        ax2.set_xlabel('Generation')

                        ax3.scatter(ii_list[:nOutput], np.log10(s_list), c='b', label='s_lmn')
                        ax4.scatter(ii_list[:nOutput], np.log10(S_signal_list[:nOutput]), c='r', label='S')
                        ax3.grid()
                        ax4.grid()
                        ax3.set_ylabel('TX-RX Local Fitness Score $\log_{10}(s_{tx,rx})$')
                        ax4.set_ylabel('Global Fitness Scor3 $\log_{10}(S)$')
                        ax3.set_xlabel('Generation')
                        ax4.set_xlabel('Generation')
                        fig.savefig(path2plots_sub + '/trend_wrt_generation.png')
                        pl.close(fig)

                        fig = pl.figure(figsize=(8, 10), dpi=120)
                        ax1 = fig.add_subplot(211)
                        ax2 = fig.add_subplot(212)
                        ax1.scatter(S_signal_list[:nOutput], s_list, c='k', label='$r_{lin} = $' + str(round(r_lin, 3)))
                        ax2.scatter(np.log10(S_signal_list[:nOutput]), np.log10(s_list), c='k',
                                    label='$r_{log} = $' + str(round(r_log, 3)))

                        ax1.grid()
                        ax2.grid()
                        ax1.set_ylabel('TX-RX Local Fitness Score $s_{tx,rx}$')
                        ax1.set_xlabel('Global Fitness Score $S$')
                        ax2.set_ylabel('TX-RX Local Fitness Score $\log_{10}(s_{tx,rx})$')
                        ax2.set_xlabel('Global Fitness Scor3 $\log_{10}(S)$')
                        ax1.legend()
                        ax2.legend()
                        fig.savefig(path2plots_sub + '/trend_wrt_global.png')
                        pl.close(fig)