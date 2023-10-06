import os.path
import sys
import numpy as np
import time
import datetime
import h5py
import configparser
from scipy.interpolate import interp1d
from makeDepthScan import depth_scan_from_hdf
import matplotlib.pyplot as pl

sys.path.append('../')

import util
import paraPropPython as ppp
from receiver import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rxList_from_file, create_transmitter_array, create_hdf_FT
from data import create_tx_signal, bscan, bscan_rxList, create_hdf_bscan

fname_nmatrix = sys.argv[1]
nmatrix_hdf = h5py.File(fname_nmatrix, 'r')
n_profile_matrix = np.array(nmatrix_hdf['n_profile_matrix'])
S_arr = np.array(nmatrix_hdf['S_arr'])
genes_arr = np.array(nmatrix_hdf['genes_matrix'])
z_genes = np.array(nmatrix_hdf['z_genes'])
nGenes = len(z_genes)
z_profile = abs(np.array(nmatrix_hdf['z_profile']))[1:-1]
nmatrix_hdf.close()

nGenerations = len(S_arr)
nGenerations_finished = 0
S_max_list = []
S_med_list = []
S_mean_list = []
gens = []
for i in range(nGenerations):
    if np.all(S_arr[i] == 0) == False:
        S_max = max(S_arr[i])
        S_mean = np.mean(S_arr[i])
        S_median = np.median(S_arr[i])
        S_max_list.append(S_max)
        S_med_list.append(S_median)
        S_mean_list.append(S_mean)
        gens.append(i)

parent_dir = os.path.dirname(fname_nmatrix)
cwd = str(os.getcwd())
print(cwd)
import subprocess

S_max_list = np.array(S_max_list)
S_mean_list = np.array(S_mean_list)
S_med_list = np.array(S_med_list)
fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
ax.plot(gens, S_max_list,c='b',label='Max')
ax.plot(gens, S_med_list, c='g',label='Median')
ax.plot(gens, S_mean_list,c='r',label='Mean')
ax.set_xlabel('Generation')
ax.legend()
ax.set_ylabel('Fitness Score $S = (\sum_{ij }\chi_{ij})^{-1} $')
ax.grid()
pl.savefig(parent_dir + '/S_score.png')
pl.close(fig)

fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
ax.plot(gens, 1/S_max_list,c='b',label='Max')
ax.plot(gens, 1/S_med_list, c='g',label='Median')
ax.plot(gens, 1/S_mean_list,c='r',label='Mean')
ax.set_xlabel('Generation')
ax.legend()
ax.set_ylabel('Global Misfit Score $\sum_{ij} \chi_{ij}$')
ax.grid()
fig.savefig(parent_dir + '/' +'M_score.png')
pl.close(fig)

all_scores = []
all_misfit = []
all_gens = []


for i in range(nGenerations-1):
    S_list = S_arr[i]
    nIndividuals = len(S_list)
    for j in range(nIndividuals):
        S_ij = S_list[j]
        M_ij = 1/S_ij
        if S_ij != 0:
            all_scores.append(S_ij)
            all_misfit.append(M_ij)
            all_gens.append(i)
#print(all_scores)
all_scores = np.array(all_scores)
all_misfit = np.array(all_misfit)
all_gens = np.array(all_gens)
N = len(all_scores) / float((nGenerations-1)*nIndividuals)
print(N)
#print(nGenerations_finished)

#H, xedges, yedges = np.histogram2d(all_gens, all_scores, bins=[nGenerations, nGenerations])
import matplotlib as mpl

'''
fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
#ax.hist2d(all_gens, all_scores, bins=[nGenerations, nGenerations], norm=mpl.colors.LogNorm())
ax.hist2d(all_gens, all_scores, bins=[nGenerations, nGenerations])

ax.plot(gens, S_max_list,c='b',label='Max')
ax.plot(gens, S_med_list, c='g',label='Median')
ax.plot(gens, S_mean_list,c='r',label='Mean')
ax.legend()
ax.set_xlabel('Generation')
ax.set_ylabel('Score')
pl.show()
'''

fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
h = ax.hist2d(all_gens, all_misfit, bins=[nGenerations, nGenerations], range=[[0, nGenerations-1],[200, 300]])
ax.plot(gens, 1/S_max_list,c='b',label='Max')
ax.plot(gens, 1/S_med_list, c='g',label='Median')
ax.plot(gens, 1/S_mean_list,c='r',label='Mean')
fig.colorbar(h[3], ax=ax)
ax.legend()
ax.set_xlabel('Generation')
ax.set_ylabel('Misfit')
pl.close(fig)
