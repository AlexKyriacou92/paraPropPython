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
import os
from genetic_functions import initialize_from_analytical, roulette, initalize_from_fluctuations, initialize
from pleiades_scripting import make_command, test_job, submit_job, test_job_data, make_command_data, make_job, countjobs
from selection_functions import selection
from genetic_functions import create_profile

sys.path.append('../')
import util
from util import get_profile_from_file, smooth_padding, do_interpolation_same_depth
from util import save_profile_to_txtfile
import matplotlib as mpl

def calculate_sij(n_prof_sim, n_prof_data):
    Nsim = len(n_prof_sim)
    Ndata = len(n_prof_data)

    dn = 0
    for i in range(Ndata):
        dn += (abs(n_prof_sim[i] - n_prof_data[i]) / n_prof_data[i])**2
    dn /= float(Nsim * Ndata)
    s = 1/dn
    return s
def calculate_S(n_prof_sim_arr, n_prof_data):
    Narr = len(n_prof_sim_arr)
    S = 0
    for i in range(n_prof_sim_arr):
        S += calculate_sij(n_prof_sim_arr[i], n_prof_data)
    S /= float(Narr)
    return S

path2_nmatrix = sys.argv[1]

hdf_nmatrix = h5py.File(path2_nmatrix, 'r')
fname_nmatrix = os.path.basename(path2_nmatrix)
n_profile_matrix = np.array(hdf_nmatrix['n_profile_matrix'])
nprof_reference = np.array(hdf_nmatrix['reference_data'])
zprof_simul = np.array(hdf_nmatrix['z_profile'])

'''
zprof_genes = np.array(hdf_nmatrix['z_genes'])
from genetic_functions import create_profile
nprof_override, zprof_override = get_profile_from_file('share/aletsch/n_prof_PS.txt')
nprof_final = create_profile(zprof_simul, nprof_reference, zprof_simul, nprof_override, zprof_override)
'''



S_arr = np.array(hdf_nmatrix['S_arr'])
hdf_nmatrix.close()
nGens = len(S_arr)
nGens_finished = 0

S_list = []
S_list_log = []
gens = []
k = 0
l = 0

S_mean = []
S_median = []
S_se = []
S_max = []

S_nprof = []
j_best = []
j_best_n = []
S_max_n = []
for i in range(nGens):
    if np.all(S_arr[i] == 0) == False:
        nGens_finished += 1
        nIndividuals = len(S_arr[i])
        S_list_n = []
        for j in range(nIndividuals):
            k += 1
            if S_arr[i,j] != 0:
                S_list_log.append(np.log10(S_arr[i,j]))
                S_list.append(S_arr[i,j])

                gens.append(i)
                l += 1

                s_ij = calculate_sij(n_prof_sim=n_profile_matrix[i,j], n_prof_data=nprof_reference)
                S_nprof.append(s_ij)
                S_list_n.append(s_ij)
        S_mean.append(np.mean(S_arr[i]))
        S_median.append(np.median(S_arr[i]))
        S_se.append(np.std(S_arr)/np.sqrt(float(nIndividuals)))
        S_max.append(max(S_arr[i]))
        j_best.append(np.argmax(S_arr[i]))
        j_best_n.append(np.argmax(S_list_n))
        S_max_n.append(max(S_list_n))
i_best_n = np.argmax(S_max_n)
i_best = np.argmax(np.argmax(S_max))
print(l, k, float(l)/float(k)*100, '%')
print(nGens_finished)
gen_lin = np.arange(0, nGens_finished, 1)
bin_sizes = (nGens_finished, nGens_finished)

print(np.corrcoef(S_nprof, S_list)[0,1])
S_nprof_log = np.log10(S_nprof)
print(np.corrcoef(S_nprof_log, S_list_log)[0,1])
fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
ax.scatter(S_nprof, S_list)
ax.grid()
pl.show()

fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
ax.scatter(np.log10(S_nprof), S_list_log)

ax.grid()
pl.show()

fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
ax.set_title(fname_nmatrix)
h=ax.hist2d(gens, S_list, bins=bin_sizes, norm=mpl.colors.LogNorm(), cmap='plasma')
ax.plot(gen_lin, S_median,c='g', label='Median')
ax.plot(gen_lin, S_max,c='b', label='Max')

ax.errorbar(gen_lin, S_mean, S_se, label='Mean +/- SE',c='k')
cbar = pl.colorbar(h[3],ax=ax)
ax.legend()
pl.show()

fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
ax.set_title(fname_nmatrix)
h=ax.hist2d(gens, S_list_log, bins=bin_sizes, norm=mpl.colors.LogNorm(), cmap='plasma')
ax.plot(gen_lin, np.log10(S_median),c='g', label='Median')
ax.plot(gen_lin, np.log10(S_max),c='b', label='Max')
ax.plot(gen_lin, np.log10(S_mean), label='Mean +/- SE',c='k')
cbar = pl.colorbar(h[3],ax=ax)
ax.legend()
pl.show()

fig = pl.figure(figsize=(5,8),dpi=120)
ax = fig.add_subplot(111)
ax.plot(nprof_reference, zprof_simul,c='k')
ax.plot(n_profile_matrix[i_best, j_best[i_best]], zprof_simul,c='b')
ax.plot(n_profile_matrix[i_best_n, j_best_n[i_best_n]], zprof_simul,c='r')

ax.set_ylim(16,0)
ax.grid()
pl.show()