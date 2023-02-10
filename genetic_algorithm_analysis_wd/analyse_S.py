import datetime
import os
import random
import sys
from math import pi

import h5py
import numpy as np
from scipy.interpolate import interp1d

from genetic_algorithm import GA, read_from_config
from makeSim_nmatrix import createMatrix
import sys

from genetic_functions import initialize_from_analytical, roulette, initalize_from_fluctuations, initialize
from pleiades_scripting import make_command, test_job, submit_job, test_job_data, make_command_data, make_job
from selection_functions import selection

import subprocess
import time
import configparser

sys.path.append('../')
import util
from matplotlib import pyplot as pl

def get_profile_from_file(fname):
    profile_data = np.genfromtxt(fname)
    z_profile = profile_data[:,0]
    n_profile = profile_data[:,1]
    return n_profile, z_profile
nprof_gul, zprof_gul = get_profile_from_file('share/guliya_reduced.txt')

fname_nmatrix = 'paraPropData/calculate_S/guliya_low_res_nmatrix.h5'
hdf_nmatrix = h5py.File(fname_nmatrix, 'r')

S_arr = np.array(hdf_nmatrix.get('S_arr'))
n_matrix = np.array(hdf_nmatrix.get('n_profile_matrix'))
z_profile = np.array(hdf_nmatrix.get('z_profile'))
hdf_nmatrix.close()
nGens = len(S_arr)

S_mean = np.zeros(nGens)
S_var = np.zeros(nGens)

nDepths = len(n_matrix[0,0])
nInds = len(n_matrix[0])
n_ave = np.ones((nGens, nDepths))
n_std = np.ones((nGens, nDepths))


for i in range(nGens):
    S_mean[i] = np.mean(S_arr[i,:])
    S_var[i] = np.std(S_arr[i,:])

    for j in range(nDepths):
        n_ave[i,j] = np.mean(n_matrix[i,:,j])
        n_std[i,j] = np.std(n_matrix[i,:,j])
    print(S_mean[i], S_var[i])

gens = np.arange(0, nGens, 1)
n_res = [0.025, 0.05, 0.075,0.1]

fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
ax.errorbar(n_res, S_mean, S_var, fmt='-o')
ax.set_xlabel('Ref Index Error $\Delta n$')
ax.set_ylabel('Fitness Score S')
ax.grid()
pl.savefig('S_score_vs_delta_n.png')
pl.close()

fig = pl.figure(figsize=(4,8),dpi=120)
ax = fig.add_subplot(111)
for i in range(nGens):
    ax.errorbar( n_ave[i], z_profile, 0.1, xerr=n_std[i], fmt='-o',label=str(n_res[i]))
ax.set_xlabel('Ref Index n')
ax.set_ylabel('Depth Z [m]')
ax.grid()
ax.legend()
ax.set_ylim(16,0)
pl.savefig('n_profile_vs_err.png')
pl.close()

fig = pl.figure(figsize=(4,8),dpi=120)
ax = fig.add_subplot(111)
for i in range(nGens):
    jj = np.random.randint(0, nInds-1)
    ax.plot(n_matrix[i,jj], z_profile,'-o',label=str(n_res[i]))
ax.plot(nprof_gul, zprof_gul, c='k',label='Truth')
ax.set_xlabel('Ref Index n')
ax.set_ylabel('Depth Z [m]')
ax.grid()
ax.legend()
ax.set_ylim(16,0)
pl.savefig('n_profile_vs_err2.png')
pl.close()


fig = pl.figure(figsize=(4,8),dpi=120)
ax = fig.add_subplot(111)
for i in range(nGens):
    jj = np.random.randint(0, nInds-1)
    ax.plot(n_matrix[i,jj]- nprof_gul, z_profile,'-o',label=str(n_res[i]))
ax.set_xlabel('Ref Index Residual $n-n_{true}$')
ax.set_ylabel('Depth Z [m]')
ax.grid()
ax.legend()
ax.set_ylim(16,0)
pl.savefig('n_profile_vs_err3.png')
pl.close()

for i in range(nGens):
    ii_gen = i
    jj_ind = 0
    cmd_prefix = 'python runSim_nProfile_from_nmatrix.py '
    fname_config = 'config_aletsch_GA_pseudo.txt'
    fname_out = 'paraPropData/calculate_S/guliya_low_res_pesudo_deltan=' + str(100*n_res[i]) + '.h5'
    cmd_i = cmd_prefix + ' ' + fname_config + ' ' + fname_nmatrix + ' ' + str(ii_gen) + ' ' + str(jj_ind) + ' ' + fname_out
    print(i, 'simulating for n_res=', n_res[i])
    os.system(cmd_i)