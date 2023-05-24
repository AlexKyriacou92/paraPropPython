import os.path
import random
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

nArgs = len(sys.argv)
nArgs_required = 3
if nArgs == nArgs_required:
    fname_nmatrix = sys.argv[1]
    path2plots = sys.argv[2]
else:
    print('nArgs =', nArgs, 'is wrong, should be:', nArgs_required)
    print('Please enter: python ',  sys.argv[0], ' <path/to/fname_nmatrix> <path/to/plot_dir>')
    sys.exit()

def get_dndz(nprof, zprof):
    zprof = abs(zprof)
    dz = abs(zprof[1] - zprof[0])
    dz_half = dz/2
    zmin = min(zprof) + dz_half
    zmax = max(zprof) - dz_half
    zprof_2 = np.arange(zmin, zmax + dz, dz)
    nDepths2 = len(zprof_2)
    dndz = np.zeros(nDepths2)
    for i in range(nDepths2):
        dn = nprof[i+1] - nprof[i]
        dndz_i = dn/dz
        dndz[i] = dndz_i
    return dndz, zprof_2


nmatrix_hdf = h5py.File(fname_nmatrix, 'r')
n_profile_matrix = np.array(nmatrix_hdf['n_profile_matrix'])
S_arr = np.array(nmatrix_hdf['S_arr'])
genes_arr = np.array(nmatrix_hdf['genes_matrix'])
z_genes = np.array(nmatrix_hdf['z_genes'])
nGenes = len(z_genes)
z_profile = abs(np.array(nmatrix_hdf['z_profile']))[1:-1]
nmatrix_hdf.close()

nGenerations = len(S_arr)
nIndividuals = len(S_arr[0])
nDepths = len(z_profile)

nGenerations_complete = 0
for i in range(nGenerations):
    S_arr_gen = S_arr[i]
    if np.all(S_arr_gen == 0) == False:
        nGenerations_complete += 1


print('z_profile',min(z_profile),max(z_profile))


if os.path.isdir(path2plots) == False:
    os.system('mkdir ' + path2plots)

nprof_error = []
nprof_list = []
S_max_list = []
genes_list = []
j_ind_list = []
for i in range(nGenerations_complete):
    S_max = max(S_arr[i])
    S_max_list.append(S_max)
    ii_max = np.argmax(S_arr[i])
    j_ind_list.append(ii_max)
    print('gen:', i, 'S_max =', S_max, 'ind:', ii_max)
    n_profile_max = n_profile_matrix[i, ii_max][1:-1]
    n_genes = genes_arr[i, ii_max]
    genes_list.append(n_genes)

    nprof_list.append(n_profile_max)

    fig = pl.figure(figsize=(5,8), dpi=100)
    ax1 = fig.add_subplot(111)
    fig.suptitle('S_max = ' + str(S_max))
    ax1.plot(n_profile_max, z_profile,c='b', label='Best Score')
    ax1.scatter(n_genes, z_genes, c='c', label='Genes')
    ii_rand = np.random.randint(0, nIndividuals-1, 1)[0]
    print(ii_rand)
    n_profile_rand = n_profile_matrix[i, ii_rand][1:-1]

    #ax1.plot(n_profile_rand, z_profile,c='r')
    ax1.grid()
    ax1.set_xlabel('Ref Index n')
    ax1.set_ylabel('Depth z [m]')
    ax1.set_ylim(16,0)
    ax1.set_xlim(1.0, 2.0)
    ax1.legend()

    fname_plot = path2plots + '/' + 'ref_index_gen' + str(i).zfill(3) + '_plot.png'
    fig.savefig(fname_plot)
    pl.close(fig)

    print('')


    print('')

i_select = np.argmax(S_max_list)
n_prof_best = nprof_list[i_select]
#n_prof_err = nprof_error[i_select]
genes_best = genes_list[i_select]
fig = pl.figure(figsize=(5,8),dpi=120)
ax = fig.add_subplot(111)
ax.plot(n_prof_best, z_profile,c='b')
dz_err = 0.2
plot_label = 'Best Profile \n Generation: ' + str(i_select) + ' ind: ' + str(j_ind_list[i_select])
ax.set_title(plot_label)
#ax.errorbar(genes_best, z_genes, dz_err*np.ones(nGenes), xerr=n_prof_err*np.ones(nGenes),color='c',label='$\Delta n = $' + str(round(n_prof_err,3)))
#ax.plot(n_profile_pseudo, z_profile_pseudo,c='k')
ax.grid()
ax.set_xlim(1.0,1.9)
ax.set_ylim(16,0)
ax.set_ylabel('Depth z [m]')
ax.set_xlabel('Ref Index n')
ax.legend()
fig.savefig(path2plots + '/' + 'n_best.png')
pl.close(fig)
#pl.show()

fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
ax.plot(S_max_list)
ax.grid()
ax.set_xlabel('Gen')
ax.set_ylabel('S')
pl.show()

fig = pl.figure(figsize=(5,8),dpi=120)
ax = fig.add_subplot(111)
ax.plot(n_prof_best**2, z_profile,c='b',label='Reconstruction (interpolation)')
dz_err = 0.2
plot_label = 'Best Profile \n Generation: ' + str(i_select) + ' ind: ' + str(j_ind_list[i_select])
ax.set_title(plot_label)
#ax.errorbar(genes_best**2, z_genes, dz_err*np.ones(nGenes), xerr=4*n_prof_err*np.ones(nGenes), fmt='o',color='c',label='$\Delta \epsilon_{r} = $' + str(round(4*n_prof_err,3)) + ' (95% CL)')
ax.grid()
ax.set_xlim(0.9,3.3)
ax.set_ylim(16,-2)
ax.set_ylabel('Depth z [m]')
ax.set_xlabel('Permittivity $\epsilon_{r}$')
ax.legend()
fig.savefig(path2plots + '/' + 'eps_best.png')
pl.close(fig)

'''
dx = 0.1
x_space = np.arange(0, 50, dx)
nX = len(x_space)
nDepths = len(z_profile)
dz = 0.05
z_space = np.arange(-10, 15+dz, dz)
nZ = len(z_space)
eps_matrix_2d = np.ones((nZ, nX))
for i in range(nX):
    i_min = util.findNearest(z_space, dz) + 1
    eps_matrix_2d[i_min:, i] = n_prof_best**2

    z_shift = np.random.uniform(-1, 1, 1)
    i_shift = int(z_shift/dz)
    if i_shift > 0:
        eps_1d =  np.roll(n_prof_best**2, i_shift)
        eps_matrix_2d[i_min:,i] = eps_1d
        eps_matrix_2d[:i_shift,i] = 1.0
    elif i_shift < 0:
        eps_1d = np.roll(n_prof_best**2, i_shift)
        eps_1d[:i_shift] = 1.0
        eps_matrix_2d[i_min:,i] = eps_1d
        eps_matrix_2d[-i_shift:, i] = n_prof_best[-1]**2
    else:
        eps_1d = n_prof_best**2
        eps_1d = eps_1d[-i_shift:] = max(n_prof_best**2)
        eps_matrix_2d[i_min:,i] = eps_1d

import cmasher as cmr

fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
ax.imshow(eps_matrix_2d, aspect='auto', extent=(0, 50,max(z_space), min(z_space)),cmap=cmr.arctic)
ax.set_ylim(max(z_space), min(z_space))
pl.show()
'''