import datetime
import os
import random
import sys
from math import pi

import subprocess
import time
import configparser

import h5py
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as pl
from genetic_algorithm import GA, read_from_config
from makeSim_nmatrix import createMatrix, createMatrix2
import sys

from genetic_functions import initialize_from_analytical, roulette, initalize_from_fluctuations, initialize
from pleiades_scripting import make_command, test_job, submit_job, test_job_data, make_command_data, make_job, countjobs
from selection_functions import selection
from fitness_function import fitness_pulse_FT_data

from genetic_functions import create_profile
from genetic_operators import flat_mutation, gaussian_mutation, fluctuation_mutation
from data import create_tx_signal
sys.path.append('../')
import util
from util import get_profile_from_file, smooth_padding, do_interpolation_same_depth
from util import save_profile_to_txtfile
from receiver import receiver
import paraPropPython as ppp

def calculate_sij(n_prof_sim, n_prof_data):
    Nsim = len(n_prof_sim)
    Ndata = len(n_prof_data)

    dn = 0
    for i in range(Ndata):
        dn += (abs(n_prof_sim[i] - n_prof_data[i]) / n_prof_data[i])**2
    dn /= float(Nsim * Ndata)
    s = 1/dn
    return s

fname_sample = 'share/aletsch_glacier_model.txt'

fname_config = 'testing/config_aletsch_GA_pseudo.txt'
config = configparser.ConfigParser()
config.read(fname_config)
GA_1 = read_from_config(fname_config=fname_config)

zMin_genes = float(config['GA']['minDepth'])
zMax_genes = float(config['GA']['maxDepth'])
iceDepth = float(config['GEOMETRY']['iceDepth'])
dz = float(config['GEOMETRY']['dz'])

nGenes = int(config['GA']['nGenes'])
zspace_genes = np.linspace(zMin_genes, zMax_genes, nGenes)
dz_genes = zspace_genes[1] - zspace_genes[0]
nprof_sample_gens = util.get_profile_from_file_decimate(fname_sample,zmin=zMin_genes,
                                                        zmax=zMax_genes,
                                                        dz_out=dz_genes)
nprof_var = 0.05*np.ones(nGenes)
fname_guliya = 'share/guliya.txt'
fname_override = 'share/aletsch/n_prof_PS.txt'
nprof_genes = util.get_profile_from_file_decimate(fname_guliya, zmin=zMin_genes, zmax=zMax_genes, dz_out=dz_genes)
nprof_gul, zprof_gul = util.get_profile_from_file_cut(fname_guliya, zmin=zMin_genes, zmax=zMax_genes)
nprof_override, zprof_override = get_profile_from_file(fname_override)
zspace_simul = np.arange(0, iceDepth+dz, dz)

nprof_psuedodata = create_profile(zspace_simul, nprof_genes, zspace_genes, nprof_override, zprof_override)


mut_thres = [0.5, 0.8, 0.85, 0.9, 0.95]
#mut_thres = [0.9]
mut_var = np.arange(0.01, 0.15, 0.02)

N1 = len(mut_var)
N2 = len(mut_thres)
sourceDepth = 11.0
dz_src = 2
maxDepth = 14
minDepth = 2
source_depths = np.arange(minDepth, maxDepth + dz_src, dz_src)
nprof_list = []
for i in range(N1):
    for j in range(N2):
        nprof_mutant_genes = fluctuation_mutation(nprof_genes, mutation_thres=mut_thres[j], mutation_var=mut_var[i])
        nprof_mutant = create_profile(zspace_simul, nprof_mutant_genes, zspace_genes, nprof_override, zprof_override)
        nprof_list.append(nprof_mutant)

N = len(nprof_list)
rx_depths = np.arange(1, 15,0.5)
rx_ranges = np.array([25, 42])
ii_select = 0
r_select = 25
z_select = sourceDepth



geometry = config['GEOMETRY']
pulse_list = []
tx_signal = create_tx_signal(fname_config)
signal_pulse = tx_signal.get_gausspulse()
dt = tx_signal.dt
tspace = tx_signal.tspace


print('done')

S_corr_list = np.ones(N)
S_diff_list = np.ones(N)
S_n_list = np.ones(N)
duration = 0
tstart = time.time()
tend = time.time()
nSrcs = len(source_depths)
print(N)

for i in range(N):
    s_ij_corr = 0
    s_ij_diff = 0
    print(i, float(i)/float(N)*100, '%')
    for l in range(nSrcs):
        sourceDepth = source_depths[l]
        ii = 0
        rxList0 = []
        print('z=', sourceDepth)
        for j in range(len(rx_ranges)):
            for k in range(len(rx_depths)):
                rx_ij = receiver(x=rx_ranges[j], z=rx_depths[k])
                ii += 1
                rxList0.append(rx_ij)
        print('Pseudo Data')
        tstart = time.time()
        sim = ppp.paraProp(iceDepth=float(geometry['iceDepth']),
                           iceLength=float(geometry['iceLength']),
                           dx=float(geometry['dx']),
                           dz=float(geometry['dz']))
        sim.set_n(nVec=nprof_list[i], zVec=zspace_simul)
        sim.set_n(nVec=nprof_psuedodata, zVec=zspace_simul)
        sim.set_dipole_source_profile(centerFreq=tx_signal.frequency, depth=sourceDepth)  # Set Source Profile
        sim.set_td_source_signal(signal_pulse, dt)  # Set transmitted signal
        sim.do_solver(rxList0, freqMin=tx_signal.freqMin, freqMax=tx_signal.freqMax)
        tend = time.time()

        duration = tend - tstart
        remainder = duration * (((N-i) * (nSrcs))*2 - l*2 - 1)
        print('Duration: ', datetime.timedelta(seconds=duration))
        print('Remaining time: ', datetime.timedelta(seconds=remainder))
        rxList = []
        for j in range(len(rx_ranges)):
            for k in range(len(rx_depths)):
                rx_ij = receiver(x=rx_ranges[j], z=rx_depths[k])
                rxList.append(rx_ij)
        print('Simuln')

        sim = ppp.paraProp(iceDepth=float(geometry['iceDepth']),
                           iceLength=float(geometry['iceLength']),
                           dx = float(geometry['dx']),
                           dz=float(geometry['dz']))
        sim.set_n(nVec=nprof_list[i], zVec=zspace_simul)
        sim.set_dipole_source_profile(centerFreq=tx_signal.frequency, depth=sourceDepth)  # Set Source Profile
        sim.set_td_source_signal(signal_pulse, dt) #Set transmitted signal

        sim.do_solver(rxList, freqMin=tx_signal.freqMin, freqMax=tx_signal.freqMax)
        for j in range(len(rxList)):
            rx_sig = rxList[j].get_signal()
            rx_sig0 = rxList0[j].get_signal()
            s_ij_corr += fitness_pulse_FT_data(sig_sim=rx_sig, sig_data=rx_sig0, mode='Correlation')
            s_ij_diff += fitness_pulse_FT_data(sig_sim=rx_sig, sig_data=rx_sig0, mode='Difference')
    s_ij_corr /= (float(len(rxList))*float(len(source_depths)))
    s_ij_diff /= (float(len(rxList)) * float(len(source_depths)))

    s_ij_n = calculate_sij(n_prof_sim=nprof_list[i], n_prof_data=nprof_psuedodata)

    S_n_list[i] = s_ij_n
    S_corr_list[i] = s_ij_corr
    S_diff_list[i] = s_ij_diff
    print('S_n', s_ij_n)
    print('s_corr', s_ij_corr)
    print('s_diff', s_ij_diff)
    print('')
r_linear_diff = np.corrcoef(S_n_list, S_diff_list)[0,1]
r_linear_corr = np.corrcoef(S_n_list, S_corr_list)[0,1]
r_log_diff = np.corrcoef(np.log10(S_n_list), np.log10(S_diff_list))[0,1]
r_log_corr = np.corrcoef(np.log10(S_n_list), np.log10(S_corr_list))[0,1]
print('Diff: r_n_A = ', r_linear_diff, ', log: ', r_log_diff)
print('Corr: r_n_A = ', r_linear_corr, ', log: ', r_log_corr)

fig = pl.figure(figsize=(10,6),dpi=120)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.set_title('Difference')
ax2.set_title('Correlation')
ax1.scatter(S_n_list, S_diff_list)
ax2.scatter(S_n_list, S_corr_list)
ax1.set_xlabel('$S_{n}$')
ax2.set_xlabel('$S_{n}$')
ax1.set_ylabel('$S_{A}$')
ax2.set_ylabel('$S_{A}$')
ax1.set_xscale('log')
ax2.set_xscale('log')
ax1.set_yscale('log')
ax2.set_yscale('log')

ax1.grid()
ax2.grid()
fig.savefig('S_between_N_A.png')
pl.show()

fig = pl.figure(figsize=(10,6),dpi=120)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.set_title('Difference')
ax2.set_title('Correlation')
ax1.scatter(S_n_list, S_diff_list)
ax2.scatter(S_n_list, S_corr_list)
ax1.set_xlabel('$S_{n}$')
ax2.set_xlabel('$S_{n}$')
ax1.set_ylabel('$S_{A}$')
ax2.set_ylabel('$S_{A}$')

ax1.grid()
ax2.grid()
fig.savefig('S_lin_between_N_A.png')
pl.show()