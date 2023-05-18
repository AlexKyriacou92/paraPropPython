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
output = str(subprocess.check_output("ls " + parent_dir + "/*pseudo*h5", shell=True))[2:-3:]
print(output)
pseudo_data = cwd + '/' + output
print(pseudo_data)
parent_size = len(parent_dir) + 1
job_label = fname_nmatrix[parent_size:-3]

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
fig.savefig(parent_dir + '/' + job_label + '_S_score.png')
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
fig.savefig(parent_dir + '/' + job_label + '_M_score.png')
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

'''
fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
ax.hist2d(all_gens, all_misfit, bins=[nGenerations, nGenerations], range=[[0, nGenerations-1],[200, 600]],
          norm=mpl.colors.LogNorm())
ax.plot(gens, 1/S_max_list,c='b',label='Max')
ax.plot(gens, 1/S_med_list, c='g',label='Median')
ax.plot(gens, 1/S_mean_list,c='r',label='Mean')
ax.legend()
ax.set_xlabel('Generation')
ax.set_ylabel('Misfit')
pl.show()
'''
i_select = 1
S_nprof_list = []
n_truth_genes = np.ones(nGenes)
bscan_pseudo = bscan_rxList()
print(type(pseudo_data))
bscan_pseudo.load_sim(pseudo_data)
n_prof_pseudo = bscan_pseudo.n_profile
z_prof_pseudo = bscan_pseudo.z_profile
S_list_all = []
for j in range(nGenes):
    j_truth = util.findNearest(z_prof_pseudo, z_genes[j])
    n_truth_genes[j] = n_prof_pseudo[j_truth]
S_nprof_max = []
gen_num = []
jj_best_nprof = []
for i_select in range(nGenerations):
    n_genes_gen = genes_arr[i_select]
    n_prof_list = []
    z_list = []
    nDepths = len(n_genes_gen[i_select])
    S_nprof_gen = np.zeros(nIndividuals)

    for i in range(len(n_genes_gen)):
        genes_ind = n_genes_gen[i]
        delta_genes = (genes_ind - n_truth_genes)**2
        Chi_genes = sum(delta_genes)
        S_nprof = 1/Chi_genes
        S_nprof_gen[i] = S_nprof
        if S_arr[i_select, i] != 0:
            S_nprof_list.append(S_nprof)
            S_list_all.append(S_arr[i_select, i])

            if S_arr[i_select, i] == S_max_list[i_select]:
                S_nprof_max.append(S_nprof)
                gen_num.append(float(i_select))
        for j in range(nDepths):
            n_prof_list.append(genes_ind[j])
            z_list.append(-z_genes[j])
    jj_best_nprof.append(np.argmax(S_nprof_gen))
    '''
    fig = pl.figure(figsize=(5,8))
    ax = fig.add_subplot(111)
    h = ax.hist2d(n_prof_list, z_list, bins=[40, 2*nDepths], range=[[1.2,1.8],[-15,-1.5]],
                  cmap='viridis_r',vmin=1, vmax=80,norm=mpl.colors.LogNorm())
    #ax.scatter(n_prof_list, z_list, c='b')
    fig.colorbar(h[3], ax=ax)
    ax.set_title('Gen = ' + str(i_select))
    ax.plot(n_prof_pseudo, -1*z_prof_pseudo, c='c')
    ax.set_label('Ref Index')
    ax.set_ylabel('Depth Z[m]')
    ax.set_ylim(-16,0)
    fig.savefig(parent_dir + '/' + 'nprof_range' + str(i_select).zfill(3) + '.png')
    pl.close(fig)

    fig = pl.figure(figsize=(5, 8))
    ax = fig.add_subplot(111)
    ax.scatter(n_prof_list, z_list, c='b',s=0.5)
    ax.set_title('Gen = ' + str(i_select))
    ax.plot(n_prof_pseudo, -1 * z_prof_pseudo, c='k')
    ax.set_label('Ref Index')
    ax.set_ylabel('Depth Z[m]')
    ax.set_ylim(-16, 0)
    ax.grid()
    ax.set_xlim(1.2,1.8)
    fig.savefig(parent_dir + '/' + 'nprof_scatter' + str(i_select).zfill(3) + '.png')
    pl.close(fig)
    '''
corr_S = np.corrcoef(S_nprof_list, S_list_all)
print(corr_S)
corr_S = np.corrcoef(S_nprof_list, np.array(S_list_all)**2)
print(corr_S)
S_list_all_sq = np.array(S_list_all)**2
S_nprof_list = np.array(S_nprof_list)
m, c = np.polyfit(S_nprof_list, S_list_all_sq, 1)
gen_num = np.array(gen_num)
gen_rel = gen_num/float(nGenerations)
import matplotlib.cm as cm
import matplotlib as mpl


norm = mpl.colors.Normalize(vmin=0, vmax=float(nGenerations))
cmap = cm.viridis
#c_arr = color_map_color(gen_num, vmin=0, vmax=float(nGenerations))
c_arr = []

x_space = np.linspace(min(S_nprof_list), max(S_nprof_list), 100)
fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
ax.scatter(S_nprof_list, np.array(S_list_all)**2,c='c',s=5)
ax.plot(S_nprof_max, np.array(S_max_list)**2,c='b')

sc = ax.scatter(S_nprof_max, np.array(S_max_list)**2,c=cmap(gen_rel))
fig.colorbar(sc)
ax.plot(x_space, m*x_space + c, label='Correlation, R = ' + str(round(corr_S[0,1],2)),c='k')
ax.set_xlabel('Profile Fitness Score $S_{n}^{2}$')
ax.set_ylabel('Waveform Fitness Score $S_{A}$')
ax.grid()
ax.legend()
fig.savefig(parent_dir + '/S_correlation.png')
pl.close(fig)

ii_best_nprof = np.argmax(S_nprof_max)
ii_best_wvf = np.argmax(S_max_list)
jj_best_wvf = np.argmax(S_arr[ii_best_wvf])

fig = pl.figure(figsize=(5,8),dpi=120)
ax = fig.add_subplot(111)
ax.plot(n_prof_pseudo, z_prof_pseudo,c='k')

ax.plot(n_profile_matrix[ii_best_nprof, jj_best_nprof[ii_best_nprof]][1:-1], z_profile,c='g',label='N-Prof Best Match')
ax.plot(n_profile_matrix[ii_best_wvf, jj_best_wvf][1:-1], z_profile,c='b',label='Wvf Best Match')

ax.grid()
ax.set_ylim(16,0)
ax.set_xlim(1.2,1.8)
ax.legend()
pl.show()