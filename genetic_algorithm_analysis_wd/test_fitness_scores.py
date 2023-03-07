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
from genetic_operators import flat_mutation, gaussian_mutation
from data import create_tx_signal
sys.path.append('../')
import util
from util import get_profile_from_file, smooth_padding, do_interpolation_same_depth
from util import save_profile_to_txtfile
from receiver import receiver
import paraPropPython as ppp
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

fig = pl.figure(figsize=(10,8),dpi=120)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

#ax.plot(nprof_gul, zprof_gul, c='k')
ax1.plot(nprof_psuedodata, zspace_simul,c='k',label='Truth')
mut_thres = [0.8, 0.85, 0.9, 0.95]
N = len(mut_thres)
sourceDepth = 11.0
nprof_list = []
for i in range(N):
    #nprof_mutant_genes = flat_mutation(nprof_genes, mutation_thres=0.95)    n
    nprof_mutant_genes = flat_mutation(nprof_genes, mutation_thres=mut_thres[i], nmin=1.3, nmax=1.6)
    #nprof_mutant_genes = gaussian_mutation(nprof_genes, mutation_thres=mut_thres[i], n_prof_mean=nprof_sample_gens, n_prof_var=nprof_var)
    nprof_mutant = create_profile(zspace_simul, nprof_mutant_genes, zspace_genes, nprof_override, zprof_override)
    nprof_list.append(nprof_mutant)
    ax1.plot(nprof_mutant, zspace_simul,label='Mutant, thres = ' + str(mut_thres[i]))
    ax2.plot(nprof_mutant-nprof_psuedodata, zspace_simul, label='Mutant, thres = ' + str(mut_thres[i]))
ax1.grid()
ax1.scatter(1.2, sourceDepth,c='b')
ax1.set_xlabel('Ref-Index n')
ax1.set_ylabel('Depth Z [m]')
ax1.legend()
ax1.set_ylim(16, 0)

ax2.grid()
ax2.set_xlabel('Ref-Index $\Delta$n')
ax2.set_ylabel('Depth Z [m]')
ax2.legend()
ax2.set_ylim(16,0)
fig.savefig('mutant_profiles.png')
pl.show()

'''for i in range(GA_1.nIndividuals):
        nprof_initial[i] = create_profile(zspace_simul, GA_1.first_generation[i], zspace_genes, nprof_override, zprof_override)
'''

rx_depths = np.arange(1, 15,0.5)
rx_ranges = np.array([25, 42])
ii_select = 0
r_select = 25
z_select = sourceDepth

ii = 0
rxList = []
for i in range(len(rx_ranges)):
    for j in range(len(rx_depths)):
        rx_ij = receiver(x=rx_ranges[i], z=rx_depths[j])
        ii += 1
        rxList.append(rx_ij)
        if rx_ranges[i] == r_select and rx_depths[j] == z_select:
            ii_select = ii
geometry = config['GEOMETRY']
pulse_list = []
tx_signal = create_tx_signal(fname_config)
signal_pulse = tx_signal.get_gausspulse()
dt = tx_signal.dt
tspace = tx_signal.tspace
fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
sim = ppp.paraProp(iceDepth=float(geometry['iceDepth']),
                       iceLength=float(geometry['iceLength']),
                       dx = float(geometry['dx']),
                       dz=float(geometry['dz']))
sim.set_n(nVec=nprof_list[i], zVec=zspace_simul)
sim.set_n(nVec=nprof_psuedodata, zVec=zspace_simul)
sim.set_dipole_source_profile(centerFreq=tx_signal.frequency, depth=sourceDepth)  # Set Source Profile
sim.set_td_source_signal(signal_pulse, dt) #Set transmitted signal
print('first solution')

sim.do_solver(rxList, freqMin=tx_signal.freqMin, freqMax=tx_signal.freqMax)
print('done')
rx_ii = rxList[ii_select]
rx_sig0 = rx_ii.get_signal()
pulse_list.append(rx_ii.get_signal())
ax.plot(tspace, rx_sig0.real,c='k',label='Original')
for i in range(N):
    rxList = []
    print(i)
    for j in range(len(rx_ranges)):
        for k in range(len(rx_depths)):
            rx_ij = receiver(x=rx_ranges[j], z=rx_depths[k])
            rxList.append(rx_ij)

    sim = ppp.paraProp(iceDepth=float(geometry['iceDepth']),
                       iceLength=float(geometry['iceLength']),
                       dx = float(geometry['dx']),
                       dz=float(geometry['dz']))
    sim.set_n(nVec=nprof_list[i], zVec=zspace_simul)
    sim.set_dipole_source_profile(centerFreq=tx_signal.frequency, depth=sourceDepth)  # Set Source Profile
    sim.set_td_source_signal(signal_pulse, dt) #Set transmitted signal

    sim.do_solver(rxList, freqMin=tx_signal.freqMin, freqMax=tx_signal.freqMax)
    rx_ii = rxList[ii_select]
    rx_sig = rx_ii.get_signal()
    pulse_list.append(rx_ii.get_signal())
    ax.plot(tspace, rx_sig.real,label=str(mut_thres[i]))
    s_ij_corr = fitness_pulse_FT_data(sig_sim=rx_sig, sig_data=rx_sig0, mode='Correlation')
    s_ij_diff = fitness_pulse_FT_data(sig_sim=rx_sig, sig_data=rx_sig0, mode='Difference')
    print('s_corr', s_ij_corr)
    print('s_diff', s_ij_diff)
    ax.plot(tspace, rx_sig.real,label=str(mut_thres[i]))

ax.grid()
ax.set_ylabel('Amplitude')
ax.set_xlabel('Time [ns]')
ax.legend()
fig.savefig('pulse_mutants.png')
pl.show()