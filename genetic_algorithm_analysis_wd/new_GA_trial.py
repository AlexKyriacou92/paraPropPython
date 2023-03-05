import datetime
import os
import random
import sys
from math import pi

import subprocess
import time
import configparser
import matplotlib as mpl

import h5py
import numpy as np
import scipy
from scipy.interpolate import interp1d
from matplotlib import pyplot as pl
from genetic_algorithm import GA, read_from_config
from makeSim_nmatrix import createMatrix
import sys

from genetic_functions import initialize_from_analytical, roulette, initalize_from_fluctuations, initialize
from genetic_functions import create_profile
from pleiades_scripting import make_command, test_job, submit_job, test_job_data, make_command_data, make_job, countjobs
from selection_functions import selection

sys.path.append('../')
import util
from util import get_profile_from_file, smooth_padding, do_interpolation_same_depth, get_profile_from_file_cut
fname_config0 = 'testing/config_aletsch_GA_pseudo.txt'
path2plots = 'testing/plots/'
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

def calculate_ave_residual(n_prof_sim, n_prof_data):
    Nsim = len(n_prof_sim)
    Ndata = len(n_prof_data)

    dn = 0
    for i in range(Ndata):
        dn += (abs(n_prof_sim[i] - n_prof_data[i]) / n_prof_data[i])
    dn /= float(Ndata)
    return dn

def dndz(n, z):
    N = len(z)
    dz = z[1]-z[0]
    dz_half = dz/2
    zmin = min(z)
    zmax = max(z)
    n0 = n[0]
    z_out = np.linspace(zmin+dz_half, zmax-dz_half, N-1)
    n_out = np.zeros(N-1)
    for i in range(N-1):
        dn = n[i+1] - n[i]
        n_out[i] = dn/dz
    return n_out, z_out, n0

def reintegrate(dndz, z_c, n0):
    dz = z_c[1]-z_c[0]
    dz_half = dz/2
    N_low = len(dndz)
    N = N_low + 1

    z_out = np.linspace(min(z_c)-dz_half, max(z_c)+dz_half, N)
    n_out = np.ones(N)
    n_out[0] = n0
    for i in range(N-1):
        n_out[i+1] = (dndz[i] + dndz[i-1])*dz/2 + n_out[i]
    return n_out

def main(fname_config):
    print('start')
    start = time.time()

    #Load Simulation Config
    print('load simulation config')
    config = configparser.ConfigParser()
    config.read(fname_config)

    #Datetime
    now = datetime.datetime.now()
    time_str = now.strftime('%y%m%d_%H%M%S')

    #Save config file
    config_cp = fname_config[:-4] + '_' + time_str + '.txt'
    os.system('cp ' + fname_config + ' ' + config_cp)
    print('saving config file to ', config_cp)

    #Set up Simulation Output Files
    print('set up output files')
    sim_mode = config['INPUT']['sim_mode']
    job_prefix = config['INPUT']['prefix']
    fname_pseudo_output0 = config['OUTPUT']['fname_pseudo_output']
    fname_nmatrix_output0 = config['OUTPUT']['fname_nmatrix_output']

    #Create directory to store results later
    results_dir = job_prefix + '_' + time_str
    #os.system('mkdir ' + results_dir)
    fname_pseudo_output = fname_pseudo_output0[:-3] + '_' + time_str + '.h5'
    fname_nmatrix_output = fname_nmatrix_output0[:-3] + '_' + time_str + '.h5'

    # Load Genetic Algorithm Properties
    print('load GA parameters')
    GA_0 = read_from_config(fname_config=config_cp)
    GA_1 = read_from_config(fname_config=config_cp)
    # Select ref-index profile to sample from
    fname_nprofile_sampling_mean = config['INPUT']['fname_sample']
    print('selecting sampling profile from: ', fname_nprofile_sampling_mean)

    # Save Sampling ref-index profiles to numpy arrays and interpolate
    print('Saving to numpy arrays')
    zMin_genes = float(config['GA']['minDepth'])
    zMax_genes = float(config['GA']['maxDepth'])
    nprof_sample_mean0, zprof_sample_mean0 = get_profile_from_file_cut(fname=fname_nprofile_sampling_mean, zmin=zMin_genes, zmax=zMax_genes)
    nprof_sample_mean, zprof_sample_mean = do_interpolation_same_depth(zprof_in=zprof_sample_mean0, nprof_in=nprof_sample_mean0, N=GA_0.nGenes)

    nStart = 1000  # Starting Sample
    dz_start = abs(zprof_sample_mean[1] - zprof_sample_mean[0])
    print('starting sample', nStart, 'dz_start = ', dz_start)


    fAnalytical = float(config['GA']['fAnalytical'])
    fFluctuations = float(config['GA']['fFluctuations'])
    fFlat = float(config['GA']['fFlat'])
    fSine = float(config['GA']['fSine'])
    fExp = float(config['GA']['fExp'])
    S_cutoff = float(config['GA']['S_cutoff'])

    #Selecting Initiate Populations
    print('initializing populations of ref-index profiles')
    n_prof_pool = initialize(nStart, nprof_sample_mean, zprof_sample_mean, GA_0, fAnalytical, fFluctuations,
                             fFlat, fSine, fExp)
    GA_0.initialize_from_sample(n_prof_pool)
    
    nprof_1st_gen = []

    fname_override = config['Override']['fname_override']
    nprof_override, zprof_override = get_profile_from_file(fname_override)


    iceDepth = float(config['GEOMETRY']['iceDepth'])
    dz = float(config['GEOMETRY']['dz'])
    zprof_simulation = np.arange(0, iceDepth+dz, dz)
    for i in range(GA_0.nIndividuals):
        nprof_genes_i = GA_0.first_generation[i]
        nprof_1st_gen_i = create_profile(zprof_simulation, nprof_genes=nprof_genes_i, zprof_genes=zprof_sample_mean,
                                         nprof_override=nprof_override, zprof_override=zprof_override)
        nprof_1st_gen.append(nprof_1st_gen_i)
    GA_1.first_generation = nprof_1st_gen

    fname_nprof_psuedodata = config['INPUT']['fname_pseudodata']

    nprof_gul0, zprof_gul0 = get_profile_from_file_cut(fname_nprof_psuedodata, zmin=zMin_genes, zmax=zMax_genes)
    dz0 = zprof_gul0[1]-zprof_gul0[0]
    dz1 = zprof_sample_mean[1] - zprof_sample_mean[0]
    M = int(round(dz1,2)/round(dz0,2))
    #sci = scipy.signal.decimate(zprof_gul0, M)
    #nprof_gul = sci(zprof_sample_mean)
    nprof_gul = scipy.signal.decimate(nprof_gul0, M)
    nprof_pseudodata = create_profile(zprof_simulation, nprof_genes=nprof_gul, zprof_genes=zprof_sample_mean,
                   nprof_override=nprof_override, zprof_override=zprof_override)
    nDepths = len(zprof_simulation)

   # nprof_matrix = np.ones((GA_1.nGenerations, GA_1.nIndividuals, nDepths))
    nprof_matrix = util.create_memmap('testing/paraPropData/nprof_matrix.npy',
                                      dimensions=(GA_1.nGenerations, GA_1.nIndividuals, nDepths),
                                      data_type='float')
    nprof_genes_matrix = np.ones((GA_1.nGenerations, GA_1.nIndividuals, GA_0.nGenes))
    S_matrix = np.zeros((GA_1.nGenerations, GA_1.nIndividuals))
    res_list = np.zeros(GA_1.nIndividuals)

    for j in range(GA_1.nIndividuals):
        nprof_genes_matrix[0,j] = GA_0.first_generation[j]
        nprof_matrix[0,j] = nprof_1st_gen[j]
        S_j = calculate_sij(n_prof_data=nprof_pseudodata, n_prof_sim=nprof_matrix[0, j])/float(GA_1.nIndividuals)
        S_matrix[0,j] = S_j
        res_list[j] = calculate_ave_residual(n_prof_data=nprof_pseudodata, n_prof_sim=nprof_matrix[0,j])
    S_max = np.zeros(GA_1.nGenerations)
    S_mean = np.zeros(GA_1.nGenerations)
    S_median = np.zeros(GA_1.nGenerations)
    S_max[0] = max(S_matrix[0])
    S_mean[0] = np.mean(S_matrix[0])
    S_median[0] = np.median(S_matrix[0])
    i_best = np.argmax(S_matrix[0])
    print('Ave Residual, <delta_n> = ', res_list[i_best]*100, '%')

    fig = pl.figure(figsize=(10, 8), dpi=120)
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.set_title('S_max = ' + str(round(S_max[0],3)))

    ax.plot(nprof_matrix[0,i_best], zprof_simulation, label='Best')
    ax.plot(nprof_pseudodata, zprof_simulation, c='k',label='Truth:\nDecimated Guliya profile + Aletsch PS data')
    # ax.set_xlim(0.8, 2.2)

    ax2.plot((nprof_matrix[0, i_best]-nprof_pseudodata)*100, zprof_simulation, c='b')
    ax2.set_ylim(16, 0)
    ax2.grid()
    ax2.set_xlabel('Ref Index Residuals $\Delta n$')

    ax.set_ylim(16, 0)
    ax.grid()
    ax.set_xlabel('Ref Index n')
    ax.set_ylabel('Depth z [m]')
    ax.legend()
    fig.savefig(path2plots +'ref-index_first_gen_results.png')
    pl.close(fig)

    nCount = 0
    np.save('testing/paraPropData/zprof_pseudodata.npy', zprof_simulation)
    np.save('testing/paraPropData/nprof_psuedodata.npy', nprof_pseudodata)

    for i in range(1, GA_1.nGenerations):
        print(i)
        n_profile_parents = nprof_genes_matrix[i-1]
        S_list = S_matrix[i-1]
        n_profile_children, inds, S_list_sorted, names = selection(prof_list=n_profile_parents, S_list=S_list,
                                               prof_list_initial=n_prof_pool,
                                               f_roulette=GA_1.fRoulette, f_elite=GA_1.fElite,
                                               f_cross_over=GA_1.fCrossOver, f_immigrant=GA_1.fImmigrant,
                                               P_mutation=GA_1.fMutation, mutation_thres=0.95)
        n_profile_children = np.array(n_profile_children)
        #print(inds, '\n', np.array(S_list_sorted)/S_list_sorted[0],'\n', names)
        res_list = np.zeros(GA_1.nIndividuals)
        for j in range(GA_1.nIndividuals):
            nprof_genes_j = n_profile_children[j]
            nprof_gen_j = create_profile(zprof_simulation, nprof_genes=nprof_genes_j, zprof_genes=zprof_sample_mean,
                                             nprof_override=nprof_override, zprof_override=zprof_override)
            nprof_matrix[i,j] = nprof_gen_j
            nprof_genes_matrix[i,j] = nprof_genes_j
            S_j = calculate_sij(n_prof_data=nprof_pseudodata,n_prof_sim=nprof_matrix[i,j])/float(GA_1.nIndividuals)
            S_matrix[i,j] = S_j
            res_list[j] = calculate_ave_residual(n_prof_data=nprof_pseudodata, n_prof_sim=nprof_matrix[i,j])
        S_max[i] = max(S_matrix[i])
        S_mean[i] = np.mean(S_matrix[i])
        S_median[i] = np.median(S_matrix[i])


        print(i, 'max(S) =', max(S_matrix[i]))
        j_best = np.argmax(S_matrix[i])
        print('Ave Residual, <delta_n> = ', res_list[j_best]*100, '%')
        if i==0 or i==5 or i % 10 == 0:
            fig = pl.figure(figsize=(10,8),dpi=120)
            ax = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax.set_title('Gen:' + str(i) + ' S_max = ' + str(round(S_max[i],3)))
            ax.plot(nprof_matrix[i, j_best], zprof_simulation, label='Best',c='b')
            ax.plot(nprof_pseudodata, zprof_simulation, c='k',label='Truth:\nDecimated Guliya profile + Aletsch PS data')
            ax2.plot((nprof_matrix[i, j_best]-nprof_pseudodata)*100, zprof_simulation, c='b')
            ax.set_ylim(16, 0)
            ax2.set_ylim(16,0)
            ax2.grid()
            ax.grid()
            ax.set_xlabel('Ref Index n')
            ax2.set_xlabel('Ref Index Residuals $\Delta n$')

            ax.set_ylabel('Depth z [m]')
            ax.legend()
            fig.savefig(path2plots +'ref-index-best-gen'+str(i)+'.png')
            pl.close(fig)
            pl.figure(figsize=(8,5),dpi=120)
            pl.plot(S_max[:nCount],c='r',label='Max')
            pl.plot(S_mean[:nCount],c='k',label='Mean')
            pl.plot(S_median[:nCount],c='g',label='Median')

            pl.grid()
            pl.xlabel('Generations')
            pl.ylabel('Score')
            pl.savefig(path2plots + 'scatter_plot.png')
            pl.close(fig)
        nCount += 1

    np.save('testing/paraPropData/S_matrix_ref_index.npy', S_matrix)
    np.save('testing/paraPropData/nprof_genes_arr.npy', nprof_genes_matrix)
    fig = pl.figure(figsize=(8, 5), dpi=120)
    pl.plot(S_max, c='r', label='Max')
    pl.plot(S_mean, c='k', label='Mean')
    pl.plot(S_median, c='g', label='Median')
    S_matrix_plot = []
    gens = []
    for k in range(GA_1.nGenerations):
        pl.scatter(float(k) * np.ones(len(S_matrix[k])), S_matrix[k], c='b')
        for l in range(GA_1.nIndividuals):
            ax.plot(S_median[:nCount], c='g', label='Median')
            gens.append(k)
            S_matrix_plot.append(S_matrix[k, l])
    pl.grid()
    pl.savefig(path2plots + 'scatter_plot.png')
    pl.close(fig)
    fig = pl.figure(figsize=(8, 5), dpi=120)
    ax = fig.add_subplot(111)
    hist2d = ax.hist2d(np.array(gens), np.array(S_matrix_plot), bins=GA_1.nGenerations)
    cbar = fig.colorbar(hist2d[3], ax=ax)
    cbar.set_label('$N_{individuals}$')
    ax.plot(S_max, c='r', label='Max')
    ax.plot(S_mean, c='k', label='Mean')
    ax.plot(S_median, c='g', label='Median')

    ax.set_xlabel('Generation')
    ax.set_ylabel('Score')
    fig.savefig(path2plots + 'S_density_map.png')
    pl.close(fig)

    fig = pl.figure(figsize=(8, 5), dpi=120)
    ax = fig.add_subplot(111)

    hist2d = ax.hist2d(np.array(gens), np.array(S_matrix_plot), norm=mpl.colors.LogNorm(), bins=nCount)

    cbar = fig.colorbar(hist2d[3], ax=ax)
    cbar.set_label('$N_{individuals}$')
    ax.plot(S_max, c='r', label='Max')
    ax.plot(S_mean, c='k', label='Mean')
    ax.plot(S_median, c='g', label='Median')

    ax.set_xlabel('Generation')
    ax.set_ylabel('Score')
    fig.savefig(path2plots + 'S_density_map_log.png')
    pl.close(fig)
    os.system('rm -f ' + config_cp)



if __name__ == '__main__':
    print('Begin Genetic Algorithm Analysis')
    main(fname_config0)
    print('GA Completed')
