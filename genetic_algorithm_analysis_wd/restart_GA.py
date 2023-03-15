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
from genetic_functions import create_profile


sys.path.append('../')
import util
from util import get_profile_from_file, smooth_padding, do_interpolation_same_depth
from util import save_profile_to_txtfile

def main(fname_config, fname_nmatrix, fname_pseudo_output, path2dir):
    config = configparser.ConfigParser()
    config.read(fname_config)
    # Step 1 -> Restart Initialization
    # Load Genetic Algorithm Properties
    print('load GA parameters')
    GA_1 = read_from_config(fname_config=fname_config)
    #GA_1 = read_from_config(fname_config=fname_config)

    # Select ref-index profile to sample from
    fname_nprofile_sampling_mean = config['INPUT']['fname_sample']
    print('selecting sampling profile from: ', fname_nprofile_sampling_mean)

    # Save Sampling ref-index profiles to numpy arrays and interpolate
    print('Saving to numpy arrays')
    # Set Overide
    zMin_genes = float(config['GA']['minDepth'])
    zMax_genes = float(config['GA']['maxDepth'])
    iceDepth = float(config['GEOMETRY']['iceDepth'])
    dz = float(config['GEOMETRY']['dz'])

    zspace_genes = np.linspace(zMin_genes, zMax_genes, GA_1.nGenes)
    dz_genes = zspace_genes[1] - zspace_genes[0]
    print(dz_genes)
    nprof_sample_mean = util.get_profile_from_file_decimate(fname=fname_nprofile_sampling_mean, zmin=zMin_genes, zmax=zMax_genes, dz_out=dz_genes)
    #TODO: -> Be careful -> Gens and dz are set indepenently -> fix this
    nStart = 10000  # Starting Sample

    fAnalytical = float(config['GA']['fAnalytical'])
    fFluctuations = float(config['GA']['fFluctuations'])
    fFlat = float(config['GA']['fFlat'])
    fSine = float(config['GA']['fSine'])
    fExp = float(config['GA']['fExp'])
    S_cutoff = float(config['GA']['S_cutoff'])
    mutation_thres = float(config['GA']['mutation_thres'])

    zspace_simul = np.arange(0, iceDepth+dz, dz)
    fname_override = config['Override']['fname_override']
    nprof_override, zprof_override = get_profile_from_file(fname_override)
    nDepths = len(zspace_simul)
    #Selecting Initiate Populations
    print('initializing populations of ref-index profiles')
    nprof_gene_pool = initialize(nStart, nprof_sample_mean, zspace_genes, GA_1, fAnalytical, fFluctuations,
                             fFlat, fSine, fExp)
    GA_1.initialize_from_sample(nprof_gene_pool)
    nprof_initial = util.create_memmap('nprof_initial.npy', dimensions=(GA_1.nIndividuals, nDepths), data_type='float')

    for i in range(GA_1.nIndividuals):
        nprof_initial[i] = create_profile(zspace_simul, GA_1.first_generation[i], zspace_genes, nprof_override, zprof_override)

    sim_mode = config['INPUT']['sim_mode']
    job_prefix = config['INPUT']['prefix']

    S_max_list = []
    S_mean_list = []
    S_var_list = []
    S_med_list = []
    gens = []

    t_max = 0
    
    if sim_mode == 'pseudo':

        cmd_prefix = 'python runSim_nProfile_pseudodata.py '
        dir_outfiles0 = path2dir + '/outfiles'

        for ii in range(nGens_complete, nGens_total):
            ii_gen = ii - 1
            # APPLY GA SELECTION
            print('Applying Selection Routines')  # TODO: Check
            gens.append(ii_gen)
            #with h5py.File(fname_nmatrix, 'r+') as nmatrix_hdf:
            nmatrix_hdf = h5py.File(fname_nmatrix, 'r+')
            S_arr = np.array(nmatrix_hdf['S_arr'])
            n_profile_matrix = nmatrix_hdf['n_profile_matrix']
            genes_matrix = nmatrix_hdf['genes_matrix']
            nprof_parents = genes_matrix[ii_gen - 1]
            S_list = np.array(S_arr[ii_gen - 1])
            print(S_list)
            if np.all(S_list == 0) == True:
                print('error, failure to run last generation, exiting')
                sys.exit()
            S_max = max(S_list)

            n_profile_children_genes = selection(prof_list=nprof_parents, S_list=S_list,
                                                 prof_list_initial=nprof_gene_pool,
                                                 f_roulette=GA_1.fRoulette, f_elite=GA_1.fElite,
                                                 f_cross_over=GA_1.fCrossOver, f_immigrant=GA_1.fImmigrant,
                                                 P_mutation=GA_1.fMutation, mutation_thres=mutation_thres)

            for j in range(GA_1.nIndividuals):
                nprof_children_genes_j = n_profile_children_genes[j]  # TODO: Check that x-y size is equal
                nprof_children_j = create_profile(zspace_simul, nprof_genes=nprof_children_genes_j,
                                                  zprof_genes=zspace_genes,
                                                  nprof_override=nprof_override,
                                                  zprof_override=zprof_override)
                n_profile_matrix[ii_gen, j] = nprof_children_j
                genes_matrix[ii_gen, j] = nprof_children_genes_j
            jj_best = np.argmax(S_arr)
            S_max_list.append(S_max)
            S_mean = np.mean(S_list)
            S_var = np.std(S_list)
            S_med = np.median(S_list)

            S_mean_list.append(S_mean)
            S_var_list.append(S_var)
            S_med_list.append(S_med)
            fname_log = path2dir + '/log_report.txt'

            with open(fname_log, 'a') as f_log:
                line = str(ii_gen) + '\t' + str(S_max) + '\t' + str(S_mean) + '\t' + str(S_var) + '\t' + str(
                    S_med) + '\n'
                f_log.write(line)

            fname_report = path2dir + '/' + 'simul_report.txt'
            if ii_gen == 0 or ii_gen == 1 or ii_gen == 5 or ii_gen % 10 == 0 or ii_gen + 1 == GA_1.nGenerations:
                with open(fname_report, 'a') as f_report:
                    line = str(ii_gen) + '\t' + str(jj_best) + '\t' + str(S_max) + '\n'
                    f_report.write(line)
            nmatrix_hdf.close()


            for j in range(GA_1.nIndividuals):
                # Create Command
                dir_outfiles = dir_outfiles0 + '/' + 'gen' + str(ii_gen)
                if os.path.isdir(dir_outfiles) == False:
                    os.system('mkdir ' + dir_outfiles)
                cmd_j = cmd_prefix + ' ' + fname_config + ' ' + fname_pseudo_output + ' ' + fname_nmatrix + ' ' + str(
                    ii_gen) + ' ' + str(j)
                jobname = job_prefix + str(ii_gen) + '-' + str(j)
                sh_file = jobname + '.sh'
                out_file = dir_outfiles + '/' + jobname + '.out'
                make_job(sh_file, out_file, jobname, cmd_j)
                submit_job(sh_file)
                # After Jobs are submitted
                os.system('rm -f ' + sh_file)


            print('jobs submitted')
            nJobs = countjobs()
            t_cycle = 0
            tsleep = 10
            if ii_gen == nGens_complete:
                while nJobs > 0:
                    print('Queue of jobs: ', nJobs)
                    print('Wait:', tsleep, ' seconds')
                    time.sleep(tsleep)
                    t_cycle += tsleep
                    print('Elapsed seconds: ', t_cycle)
                    print('Elapsed time: ', datetime.timedelta(seconds=t_cycle))
                    nJobs = countjobs()
                    print('Queue of jobs:', nJobs)
                t_max = 3*t_cycle
            else:
                while nJobs > 0 and t_cycle < t_max:
                    print('Queue of jobs: ', nJobs)
                    print('Wait:', tsleep, ' seconds')
                    time.sleep(tsleep)
                    t_cycle += tsleep
                    print('Elapsed seconds: ', t_cycle)
                    print('Elapsed time: ', datetime.timedelta(seconds=t_cycle))
                    print('Queue of jobs:', nJobs)

                    nJobs = countjobs()
                    print('')
                if t_cycle > t_max and nJobs > 0:
                    nAttempts = 5
                    for k in range(nAttempts):
                        os.system('python kill-jobs.py')
                        t_sleep2 = 60
                        time.sleep(t_sleep2)
                        print('Killing jobs, sleep for', datetime.timedelta(seconds=t_sleep2))
                        if nJobs == 0:
                            break

if len(sys.argv) == 2:
    path2dir = sys.argv[1]
    fname_config = path2dir + 'config-file.txt'
    if os.path.isfile(fname_config) == False:
        print('error, config file', fname_config, 'does not exist')
        sys.exit()
elif len(sys.argv) == 3:
    path2dir = sys.argv[1]
    fname_config = sys.argv[2] #Override Config File, must include full path!
    if os.path.isfile(fname_config) == False:
        print('error, override config file', fname_config, 'does not exist')
        sys.exit()
else:
    print('error! Must specific path to directory, arg number must be: 1 or 2')
    print('python ', sys.argv[0], ' <path2simul> <new_configfile.txt?>')
    sys.exit()
    
fname_report = path2dir + 'simul_report.txt'
if os.path.isfile(fname_report) == True:
    with open(fname_report, 'r') as f_report:
        next(f_report)
        print(f_report.readline())
        next(f_report)
        print(f_report.readline())
        line0 = f_report.readline()
        col_files = line0.split()
        fname_pseudo_output_suffix = col_files[0]
        fname_pseudo_output = path2dir + fname_pseudo_output_suffix
        if os.path.isfile(fname_pseudo_output) == False:
            print('error, ', fname_pseudo_output, 'does not exist')
            sys.exit()
        fname_nmatrix_suffix = col_files[1]
        fname_nmatrix = path2dir + fname_nmatrix_suffix
        if os.path.isfile(fname_nmatrix) == False:
            print('error,', fname_nmatrix, 'does not exist')
            sys.exit()
else:
    print('error, ', fname_report, 'does not exist')
    sys.exit()
    

#with h5py.File(fname_nmatrix) as nmatrix_hdf:
print('Calculate Completed Generations')
nmatrix_hdf = h5py.File(fname_nmatrix, 'r')
S_arr = nmatrix_hdf['S_arr']
nGens_total = len(S_arr)
nGens_complete = 0
for i in range(nGens_total):
    S_arr_gen = S_arr[i]
    if np.all(S_arr_gen == 0) == False:
        nGens_complete += 1
    else:
        break
print('Generations Completed: ', nGens_complete)
nmatrix_hdf.close()

if __name__ == '__main__':
    print('Begin Genetic Algorithm Analysis')
    main(fname_config=fname_config, fname_nmatrix=fname_nmatrix, fname_pseudo_output=fname_pseudo_output, path2dir=path2dir)
    print('GA Completed')