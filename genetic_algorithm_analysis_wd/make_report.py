import numpy as np
import sys
sys.path.append('../')

import paraPropPython as ppp
from receiver import receiver
from data import bscan, bscan_rxList

import scipy.signal as signal
import util
import h5py
import matplotlib.pyplot as pl
import peakutils as pku
import sys
import os
import configparser
from scipy.interpolate import interp1d

#fname_pseudo = sys.argv[1]
#fname_report = sys.argv[2]
fname_report = sys.argv[1]

#fname_report = pathto_report + 'simul_report.txt'
#fname_sim = sys.argv[2]
#path2report = sys.argv[3]


def ascan_plot(tspace, bscan_pseudo, bscan_sim, z_tx, path2imgs, plot_label = '', mode = 'pulse'):
    fig = pl.figure(figsize=(8, 5), dpi=200)
    ax1 = fig.add_subplot(111)
    ax1.set_title(plot_label)
    if mode == 'pulse':
        ax1.plot(tspace, bscan_pseudo.real, c='b', label='Psuedo-Data')
        ax1.plot(tspace, bscan_sim.real, c='r', label='Best results', alpha=0.5)
        ax1.grid()
        ax1.legend()
        fname_plot = path2imgs + 'ascan_pulse-z=' + str(round(z_tx, 1)) + 'm.png'
    elif mode == 'abs':
        ax1.plot(tspace, abs(bscan_pseudo), c='b', label='Psuedo-Data')
        ax1.plot(tspace, abs(bscan_sim), c='r', label='Best results', alpha=0.5)
        ax1.grid()
        ax1.legend()
        fname_plot = path2imgs + 'ascan_abs-z=' + str(round(z_tx, 1)) + 'm.png'
    elif mode == 'power':
        ax1.plot(tspace, abs(bscan_pseudo)**2, c='b', label='Psuedo-Data')
        ax1.plot(tspace, abs(bscan_sim)**2, c='r', label='Best results', alpha=0.5)
        ax1.grid()
        ax1.legend()
        fname_plot = path2imgs + 'ascan_power-z=' + str(round(z_tx, 1)) + 'm.png'
    elif mode == 'dB':
        ax1.plot(tspace, 20*np.log10(abs(bscan_pseudo)), c='b', label='Psuedo-Data')
        ax1.plot(tspace, 20*np.log10(abs(bscan_sim)), c='r', label='Best results', alpha=0.5)
        ax1.grid()
        ax1.legend()
        fname_plot = path2imgs + 'ascan_dB-z=' + str(round(z_tx, 1)) + 'm.png'
    pl.savefig(fname_plot)
    pl.close(fig)

def double_bscan_plot(bscan_plot_sim, bscan_plot_pseudo, tspace, tx_depths, path2imgs, plot_label = 'Paralell Bscan [dBu]', vmin0 = -90, vmax0= -10):
    fig = pl.figure(figsize=(8, 6), dpi=200)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    fig.suptitle(plot_label)
    ax1.set_title('Simulation')
    pmesh = ax1.imshow(10 * np.log10((abs(bscan_plot_sim) ** 2)), aspect='auto',
                       extent=[0, tspace[-1], tx_depths[-1], tx_depths[0]], vmin=vmin0, vmax=vmax0)
    # cbar = pl.colorbar(pmesh)
    # cbar.set_label('Power P [dBu]')
    ax1.set_ylabel(r'Depth $Z_{tx}$ [m]')
    ax1.set_xlabel(r'$t$ [ns]')
    ax1.set_ylim(15, 1)

    ax2.set_title('Pseudo-Data')
    pmesh = ax2.imshow(10 * np.log10((abs(bscan_plot_pseudo) ** 2)), aspect='auto',
                       extent=[0, tspace[-1], tx_depths[-1], tx_depths[0]], vmin=vmin0, vmax=vmax0)
    # cbar = pl.colorbar(pmesh)
    # cbar.set_label('Power P [dBu]')
    ax2.set_ylabel(r'Depth $Z_{tx}$ [m]')
    ax2.set_xlabel(r'$t$ [ns]')
    ax2.set_ylim(15, 1)

    pl.savefig(path2imgs + '/double-bscan.png')
    pl.close()
def single_bscan_plot(bscan_plot, tspace, tx_depths, path2imgs, plot_label = 'Paralell Bscan [dBu]', vmin0 = -90, vmax0= -10):
    fig = pl.figure(figsize=(6, 6), dpi=200)
    ax = fig.add_subplot(111)
    ax.set_title(plot_label)
    pmesh = ax.imshow(10 * np.log10((abs(bscan_plot_sim) ** 2)), aspect='auto',
                       extent=[0, tspace[-1], tx_depths[-1], tx_depths[0]], vmin=vmin0, vmax=vmax0)
    cbar = pl.colorbar(pmesh)
    cbar.set_label('Power P [dBu]')
    ax.set_ylabel(r'Depth $Z_{tx}$ [m]')
    ax.set_xlabel(r'$t$ [ns]')
    ax.set_ylim(15, 1)

    pl.savefig(path2imgs + '/bscan.png')
    pl.close()

mode_options = ['pulse', 'abs', 'power', 'dB']
nModes = len(mode_options)

f_report = open(fname_report,'r')
next(f_report)
line_2 = f_report.readline()
cols_2 = line_2.split()
path2report = cols_2[0]

report_dir0 = path2report + 'report'
if os.path.isdir(report_dir0) == False:
    os.system('mkdir ' + report_dir0)

nOutput = int(cols_2[1])
next(f_report)
next(f_report)
line_3 = f_report.readline()
cols_3 = line_3.split()
print(cols_3)
fname_pseudo = path2report + cols_3[0]
fname_nmatrix = path2report + cols_3[1]
print(fname_pseudo, fname_nmatrix)
next(f_report)
#next(f_report)

#Plot n-matrix
hdf_nmatrix = h5py.File(fname_nmatrix, 'r')

n_profile_matrix = np.array(hdf_nmatrix.get('n_profile_matrix'))
S_nmatrix = np.array(hdf_nmatrix.get('S_arr'))
z_profile = np.array(hdf_nmatrix.get('z_profile'))

nGenerations = int(hdf_nmatrix.attrs['nGenerations'])
nIndividuals = int(hdf_nmatrix.attrs['nIndividuals'])
rxList = np.array(hdf_nmatrix.get('rxList'))
signalPulse = np.array(hdf_nmatrix.get('signalPulse'))
tspace = np.array(hdf_nmatrix.get('tspace'))
source_depths = np.array(hdf_nmatrix.get('source_depths'))

fname_config = path2report + hdf_nmatrix.attrs['config_file']
print(fname_config)
config = configparser.ConfigParser()
config.read(fname_config)
print(os.path.isfile(fname_config))

input_config = config['INPUT']
print(input_config)
fname_txt = config['INPUT']['fname_pseudodata'] #File name containing profile data

hdf_nmatrix.close()

S_best = np.ones(nGenerations)
gens = np.arange(1, nGenerations+1, 1)

best_individuals = []

profile_data = np.genfromtxt(fname_txt)
nprof_data = profile_data[:,1]
zprof_data = profile_data[:,0]

S_median = np.zeros(nGenerations)
S_mean = np.zeros(nGenerations)
S_variance = np.zeros(nGenerations)
for i in range(nGenerations):
    S_best[i] = max(S_nmatrix[i])
    best_individuals.append(np.argmax(S_nmatrix[i]))
    S_median[i] = np.median(S_nmatrix[i])
    S_mean[i] = np.mean(S_nmatrix[i])
    S_variance[i] = np.std(S_nmatrix[i])
    print(S_variance[i]/S_mean[i])

ii_best_gen = np.argmax(S_best)
jj_best_ind = best_individuals[ii_best_gen]
n_profile_best = n_profile_matrix[ii_best_gen, jj_best_ind]
fig = pl.figure(figsize=(8,5),dpi=120)
pl.plot(gens, S_best,c='b')
pl.xlabel('Generation')
pl.ylabel(r'Best score $S_{max}$')
pl.grid()
pl.savefig(path2report + 'report/S_best_result.png')
pl.close()

fig = pl.figure(figsize=(8,5),dpi=120)
pl.errorbar(gens, S_mean, S_variance,fmt='-o',c='k',label='Mean +/- Variance')
pl.plot(gens, S_best,c='b',label='Best Score')
pl.plot(gens, S_median,c='r', label='Median')

pl.xlabel('Generation')
pl.ylabel(r'Fitness Score $S$')
pl.grid()
pl.legend()
pl.savefig(path2report + 'report/S_evolution.png')
pl.close()

fig = pl.figure(figsize=(4,10),dpi=120)
pl.title(r'Generation: '+ str(ii_best_gen) + r', Individual: ' + str(jj_best_ind) + r', S = ' + str(round(S_best[ii_best_gen]/1e6,2)) + r' $ \times 10^{6}$')
pl.plot(n_profile_best, z_profile, '-o', c='b',label='Best Result')
pl.plot(nprof_data, zprof_data,c='k',label='truth')
pl.grid()
pl.ylim(16,-1)
pl.ylabel(r'Depth z [m]')
pl.xlabel(r'Refractive Index Profile $n(z)$')
pl.legend()
pl.savefig(path2report + 'report/nprof_best_result.png')
pl.close()

f_interp_data = interp1d(zprof_data, nprof_data)
z_space = np.linspace(min(z_profile), max(z_profile), len(z_profile))
ii_min = util.findNearest(zprof_data, min(z_profile))
ii_max = util.findNearest(zprof_data, max(z_profile))
n_space_interp = np.ones(len(z_profile))
n_space_interp[0] = nprof_data[ii_min]
n_space_interp[-1] = nprof_data[ii_max]
n_space_interp[1:-1] = f_interp_data(z_profile[1:-1])

n_residuals = n_profile_best - n_space_interp

fig = pl.figure(figsize=(4,10),dpi=120)
pl.title(r'Generation: '+ str(ii_best_gen) + r', Individual: ' + str(jj_best_ind) + r', S = ' + str(round(S_best[ii_best_gen]/1e6,2)) + r' $ \times 10^{6}$')
pl.plot(n_residuals, z_profile, '-o', c='b',label='Residuals')
pl.axvline(0,c='r')
pl.fill_betweenx(z_profile, -0.025, +0.025,color='r',alpha=0.5,label='Boundary +/- 2.5%')
pl.grid()
pl.ylim(16,-1)
pl.ylabel(r'Depth z [m]')
pl.xlabel(r'Refractive Index Profile (residuals) $\Delta n(z)$')
pl.legend()
pl.savefig(path2report + 'report/nprof_best_result_residuals.png')
pl.close()

#======================================================

sim_bscan_pseudo = bscan_rxList()
sim_bscan_pseudo.load_sim(fname_pseudo)
tspace = sim_bscan_pseudo.tx_signal.tspace
tx_depths = sim_bscan_pseudo.tx_depths
nSamples = sim_bscan_pseudo.nSamples
nDepths = len(tx_depths)
rxList = sim_bscan_pseudo.rxList
print(rxList[0].x)
rx_depths = []
for i in range(len(rxList)):
    rx_depths.append(rxList[i].z)

bscan_sig_pseudo = sim_bscan_pseudo.bscan_sig
bscan_plot_psuedo = np.zeros((nDepths, nSamples), dtype='complex')

for line in f_report:
    cols = line.split()
    ii_gen = int(cols[0])
    jj_ind = int(cols[1])
    S_value = float(cols[2])

    fname_sim = cols[3]
    sim_dir = fname_sim[:-3]

    nprof_ij = n_profile_matrix[ii_gen, jj_ind]

    sim_bscan_GA = bscan_rxList()
    sim_bscan_GA.load_sim(path2report + fname_sim)
    bscan_sig_sim = sim_bscan_GA.bscan_sig
    bscan_plot_sim = np.zeros((nDepths, nSamples), dtype='complex')

    report_dir = path2report + 'report/' + sim_dir + '_report'
    if os.path.isdir(report_dir) == False:
        os.system('mkdir ' + report_dir)

    fig = pl.figure(figsize=(4, 10), dpi=120)
    pl.title(r'Generation: ' + str(ii_gen) + r', Individual: ' + str(jj_ind) + r', S = ' + str(
        round(S_value, 2)))
    pl.plot(nprof_ij, z_profile, '-o', c='b', label='Best Result from generation')
    pl.plot(nprof_data, zprof_data, c='k', label='truth')
    pl.grid()
    pl.ylim(16, -1)
    pl.ylabel(r'Depth z [m]')
    pl.xlabel(r'Refractive Index Profile $n(z)$')
    pl.legend()
    pl.savefig(path2report + 'report/nprof_' + sim_dir + '.png')
    pl.savefig(report_dir + '/nprof_' + sim_dir + '.png')
    pl.close()

    for i in range(nDepths):
        print('gen:', ii_gen, 'ind:', jj_ind, 'S =', S_value)
        print(i, 'tx =', tx_depths[i])
        plot_label = 'gen: ' + str(ii_gen) + ' ind: ' + str(jj_ind) + ' S = ' + str(round(S_value,3)) + '\n'
        plot_label += 'z = ' + str(tx_depths[i]) + ' m'

        jj_rx = util.findNearest(rx_depths, tx_depths[i])
        bscan_plot_psuedo[i] = bscan_sig_pseudo[i, jj_rx]
        bscan_plot_sim[i] = bscan_sig_sim[i, jj_rx]

        for j in range(nModes):
            plot_dir = report_dir + '/' + mode_options[j] + '/'
            if os.path.isdir(plot_dir) == False:
                os.system('mkdir ' + plot_dir)
            ascan_plot(tspace=tspace, bscan_pseudo=bscan_plot_psuedo[i], bscan_sim=bscan_plot_sim[i], z_tx= tx_depths[i], path2imgs=plot_dir, plot_label=plot_label, mode=mode_options[j])
    plot_label_bscan = input_config['fname_pseudodata'] + '\ngen: ' + str(ii_gen) + ' ind: ' + str(
        jj_ind) + ' S = ' + str(round(S_value, 3))

    double_bscan_plot(bscan_plot_sim=bscan_plot_sim, bscan_plot_pseudo=bscan_plot_psuedo, tspace=tspace,
                      plot_label=plot_label_bscan, tx_depths=tx_depths, path2imgs=report_dir)
    plot_label_bscan += '\nParallel Bscan [dBu]'
    single_bscan_plot(bscan_plot=bscan_plot_sim, tspace=tspace, tx_depths=tx_depths, path2imgs=report_dir,
                      plot_label=plot_label_bscan)
plot_label_pseudo = input_config['fname_pseudodata'] + '\nPseudo-Data Bscan'
single_bscan_plot(bscan_plot=bscan_plot_psuedo, tspace=tspace, tx_depths=tx_depths,
                  path2imgs=path2report,plot_label=plot_label_pseudo)