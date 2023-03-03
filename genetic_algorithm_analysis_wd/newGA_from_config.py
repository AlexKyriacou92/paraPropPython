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
nArgs = len(sys.argv)
if nArgs != 2:
    print('error, incorrect number of arguments, nArgs = ', nArgs)
    print('enter: python ', sys.argv[0], ' <fname_config>')
    sys.exit()

fname_config0 = sys.argv[1]

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
    os.system('mkdir ' + results_dir)
    fname_pseudo_output = fname_pseudo_output0[:-3] + '_' + time_str + '.h5'
    fname_nmatrix_output = fname_nmatrix_output0[:-3] + '_' + time_str + '.h5'

    # Load Genetic Algorithm Properties
    print('load GA parameters')
    GA_1 = read_from_config(fname_config=config_cp)
    #GA_1 = read_from_config(fname_config=config_cp)

    # Select ref-index profile to sample from
    fname_nprofile_sampling_mean = config['INPUT']['fname_sample']
    print('selecting sampling profile from: ', fname_nprofile_sampling_mean)

    # Save Sampling ref-index profiles to numpy arrays and interpolate
    print('Saving to numpy arrays')
    nprof_sample_mean0, zprof_sample_mean0 = get_profile_from_file(fname_nprofile_sampling_mean)
    nprof_sample_mean, zprof_sample_mean = do_interpolation_same_depth(zprof_in=zprof_sample_mean0, nprof_in=nprof_sample_mean0, N=GA_1.nGenes)

    nStart = 10000  # Starting Sample
    dz_start = abs(zprof_sample_mean[1] - zprof_sample_mean[0])
    print('starting sample', nStart, 'dz_start = ', dz_start)


    fAnalytical = float(config['GA']['fAnalytical'])
    fFluctuations = float(config['GA']['fFluctuations'])
    fFlat = float(config['GA']['fFlat'])
    fSine = float(config['GA']['fSine'])
    fExp = float(config['GA']['fExp'])
    S_cutoff = float(config['GA']['S_cutoff'])
    mutation_thres = float(config['GA']['mutation_thres'])

    # Set Overide
    zMin_genes = float(config['GA']['minDepth'])
    zMax_genes = float(config['GA']['maxDepth'])
    iceDepth = float(config['GEOMETRY']['iceDepth'])
    dz = float(config['GEOMETRY']['dz'])

    zspace_genes = np.linspace(zMin_genes, zMax_genes, GA_1.nGenes)
    dz_genes = zspace_genes[1]-zspace_genes[0]

    zspace_simul = np.arange(0, iceDepth+dz, dz)
    fname_override = config['Override']['fname_override']
    nprof_override, zprof_override = get_profile_from_file(fname_override)
    nDepths = len(zspace_simul)
    #Selecting Initiate Populations
    print('initializing populations of ref-index profiles')
    nprof_gene_pool = initialize(nStart, nprof_sample_mean, zprof_sample_mean, GA_1, fAnalytical, fFluctuations,
                             fFlat, fSine, fExp)
    GA_1.initialize_from_sample(nprof_gene_pool)
    nprof_initial = util.create_memmap('nprof_initial.npy', dimensions=(GA_1.nIndividuals, nDepths), data_type='float')

    for i in range(GA_1.nIndividuals):
        nprof_initial[i] = create_profile(zspace_simul, GA_1.first_generation[i], zspace_genes, nprof_override, zprof_override)

    #Create N-Matrix
    print('creating output matrix for storing ref-index profiles (each generation)')
    """
    The N-Matrix refers to a .h5 file which simultenenously stores fitness scores S in an array
    and also stores to ref-index profiles (individuals in each generatino) as a matrix
    with a matrid of dimensions nGenes x nIndividuals x nGenerations (nGenes is the number ref-index/depth values
    """
    fname_nmatrix = fname_nmatrix_output
    if os.path.isfile(fname_nmatrix) == True:
        os.system('rm -f ' + fname_nmatrix)
    createMatrix2(fname_config=config_cp, n_prof_initial=nprof_initial,
                  genes_initial=GA_1.first_generation,
                  z_genes=zspace_genes, z_profile=zspace_simul,
                  fname_nmatrix=fname_nmatrix, nGenerations=GA_1.nGenerations)
    hdf_nmatrix = h5py.File(fname_nmatrix, 'r+')
    hdf_nmatrix.attrs['datetime'] = time_str
    hdf_nmatrix.attrs['config_file'] = config_cp
    hdf_nmatrix.close()

    print('selecting simulation mode:', sim_mode)
    dir_outfiles0 = results_dir + '/outfiles'
    os.system('mkdir ' + dir_outfiles0)


    if sim_mode == 'pseudo':
        fname_pseudodata0 = config['INPUT']['fname_pseudodata']

        # Create Pseudo_Data Profile
        nprof_genes, zprof_genes = util.get_profile_from_file_decimate(fname=fname_pseudodata0,
                                                                                   zmin=zMin_genes, zmax=zMax_genes,
                                                                                   dz_out=dz_genes)
        nprof_pseudodata = create_profile(zspace_simul, nprof_genes=nprof_genes, zprof_genes=zprof_genes,
                                          nprof_override = nprof_override, zprof_override=zprof_override)
        fname_pseudodata = fname=dir_outfiles0 + 'nprof_pseudodata.txt'
        save_profile_to_txtfile(zprof=zspace_simul,nprof=nprof_pseudodata, fname=fname_pseudodata)

        # Create Pseudo_Data Bscan
        print('create pseudo data')
        if os.path.isfile(fname_pseudo_output) == True:
            os.system('rm -f ' + fname_pseudo_output)
        cmd = 'python runSim_pseudodata_from_txt.py ' + config_cp + ' ' + fname_pseudodata + ' ' + fname_pseudo_output
        os.system(cmd)

        hdf_nmatrix = h5py.File(fname_nmatrix, 'r+')
        profile_data = np.genfromtxt(fname_pseudodata)
        hdf_nmatrix.create_dataset('reference_data',data=nprof_pseudodata)
        hdf_nmatrix.close()

        # Next -> Calculate list of S parameters
        ii_gen = 0  # Zeroth Generation
        cmd_prefix = 'python runSim_nProfile_pseudodata.py '

        print('calculate S for individuals in 1st generation')
        outfile_list = []
        for j in range(GA_1.nIndividuals):
            dir_outfiles = dir_outfiles0 + '/' + 'gen' + str(ii_gen)
            if os.path.isdir(dir_outfiles) == False:
                os.system('mkdir ' + dir_outfiles)
            cmd_j = cmd_prefix + ' ' + config_cp + ' ' + fname_pseudo_output + ' ' + fname_nmatrix + ' ' + str(
                ii_gen) + ' ' + str(j)
            jobname = job_prefix + str(ii_gen) + '-' + str(j)
            sh_file = jobname + '.sh'
            out_file = dir_outfiles + '/' + jobname + '.out'
            outfile_list.append(out_file)
            print(out_file)
            make_job(sh_file, out_file, jobname, cmd_j)
            submit_job(sh_file)
            os.system('rm -f ' + sh_file)

        print('jobs submitted')
        proceed_bool = False
        tstart_1st_gen = time.time()
        while proceed_bool == False:
            tsleep = 10.
            nJobs = countjobs()
            print(nJobs, 'jobs left, wait:', tsleep)
            if nJobs > 0:
                time.sleep(tsleep)
            else:
                proceed_bool = True
        tend_1st_gen = time.time()
        duration_1st_gen = tend_1st_gen-tstart_1st_gen

        ii_gen += 1
        # Wait for jobs to be submitted
        print('1st generation finished')
        print('next generation:')

        S_max = 0
        S_max_list = []
        S_mean_list = []
        S_var_list = []
        S_med_list = []

        fname_log = results_dir + 'log_report.txt'
        f_log = open(fname_log,'w')
        f_log.write('gen\tS_max\tS_mean\tS_var\tS_med\n')

        tsleep = 10.
        max_time = 2 * duration_1st_gen
        while (ii_gen < GA_1.nGenerations) or (S_max < S_cutoff):
            nJobs = countjobs()
            print('Generation', ii_gen, 'Check jobs')
            t_cycle = 0

            if (nJobs == 0) or (t_cycle > max_time):
                print('Submit Jobs Now \n')

                #APPLY GA SELECTION
                print('Applying Selection Routines')
                nmatrix_hdf = h5py.File(fname_nmatrix, 'r+')
                S_arr = np.array(nmatrix_hdf['S_arr'])
                n_profile_matrix = nmatrix_hdf['n_profile_matrix']
                genes_matrix = nmatrix_hdf['genes_matrix']
                nprof_parents = genes_matrix[ii_gen-1]
                S_list = np.array(S_arr[ii_gen - 1])
                S_max = max(S_list)
                print(ii_gen - 1)
                n_profile_children_genes = selection(prof_list=nprof_parents, S_list=S_list,
                                               prof_list_initial=nprof_initial,
                                               f_roulette = GA_1.fRoulette,  f_elite = GA_1.fElite,
                                               f_cross_over = GA_1.fCrossOver, f_immigrant = GA_1.fImmigrant,
                                               P_mutation = GA_1.fMutation, mutation_thres = mutation_thres)
                n_profile_children_genes = np.array(n_profile_children_genes)
                n_profile_matrix[ii_gen] = n_profile_matrix
                for j in range(GA_1.nIndividuals):
                    nprof_children_genes_j = n_profile_children_genes[j]
                    nprof_children_j = create_profile(zspace_simul, nprof_genes=nprof_children_genes_j,
                                                    zprof_genes=zspace_genes,
                                                    nprof_override=nprof_override,
                                                    zprof_override=zprof_override)
                    n_profile_matrix[ii_gen, j] = nprof_children_j
                    genes_matrix[ii_gen, j] = nprof_children_genes_j
                S_max_list.append(S_max)
                S_mean = np.mean(S_list)
                S_var = np.std(S_list)
                S_med = np.mean(S_list)
                S_mean_list.append(S_mean)
                S_var_list.append(S_var)
                S_med_list.append(S_med)
                gens = np.arange(0, ii_gen, 1)
                # Make Plots:

                fig = pl.figure(figsize=(8,5),dpi=120)
                ax = fig.add_subplot(111)
                ax.errorbar(gens, S_mean_list, S_var_list, fmt='-o', c='k', label='Mean +/- Variance')
                ax.plot(gens, S_max_list, c='b', label='Best Score')
                ax.plot(gens, S_med_list, c='r', label='Median')
                ax.set_xlabel('Generation')
                ax.set_ylabel(r'Fitness Score $S$')
                ax.grid()
                ax.legend()
                pl.savefig(results_dir + '/' + 'S_current.png')
                pl.close(fig)

                line = str(ii_gen) + '\t' + str(S_max) + '\t' + str(S_mean) + '\t' + str(S_var) + '\t' + str(S_med) + '\n'
                f_log.write(line)
                for j in range(GA_1.nIndividuals):
                    #Create Command
                    dir_outfiles = dir_outfiles0 + '/' + 'gen' + str(ii_gen)
                    if os.path.isdir(dir_outfiles) == False:
                        os.system('mkdir ' + dir_outfiles)
                    cmd_j = cmd_prefix + ' ' + config_cp + ' ' + fname_pseudo_output + ' ' + fname_nmatrix + ' ' + str(
                        ii_gen) + ' ' + str(j)
                    jobname = job_prefix + str(ii_gen) + '-' + str(j)
                    sh_file = jobname + '.sh'
                    out_file = dir_outfiles + '/' + jobname + '.out'
                    outfile_list.append(out_file)
                    make_job(sh_file, out_file, jobname, cmd_j)
                    submit_job(sh_file)
                    # After Jobs are submitted
                    os.system('rm -f ' + sh_file)
                ii_gen += 1
                print('Jobs running -> Generation: ', ii_gen)
            else:
                print('Queue of jobs: ', nJobs)
                print('Wait:', tsleep, ' seconds')
                time.sleep(tsleep)
                t_cycle += tsleep
        f_log.close()
        for k in range(len(outfile_list)):
            out_file_k = outfile_list[k]
            os.system('rm -f ' + out_file_k)

    else:
        #print('error, incorrect sim_mode, enter: pseudo or data')
        print('error, incorrect sim_mode, enter: pseudo')
        sys.exit()
    '''
    #TODO: Choose Data
    elif sim_mode == 'data':
        fname_data = config['INPUT']['fname_data']
        ii_gen = 0  # Zeroth Generation
        cmd_prefix = 'python runSim_nProfile_FT.py '
    '''
    # Final Step -> mv

    os.system('mv ' + config_cp + ' ' + fname_pseudo_output + ' ' + fname_nmatrix_output + ' ' + results_dir + '/')

    print('Simulating Bscans of Best Results')
    cmd_sim_best = 'python simulate_best_results.py ' + results_dir + '/' + config_cp + ' '
    cmd_sim_best += results_dir + '/' + fname_nmatrix_output + ' '
    cmd_sim_best += results_dir + '/' + fname_pseudo_output + ' '
    cmd_sim_best += results_dir + '/'
    cmd_sim_best += str(10)
    print('running:', cmd_sim_best)
    os.system(cmd_sim_best)

    print('Making Report (plots)')
    cmd_make_report = 'python make_report.py ' + results_dir + '/' + 'simul_report.txt'
    print('running: ', cmd_make_report)
    os.system(cmd_make_report)
    now = datetime.datetime.now()

    #tend = time.time()
    #duration_s = tend - tstart
    #print('Finished at: ', now.strftime('%y.%m.%d %H:%M:%S'))
    #print('Total run time: ', datetime.timedelta(seconds=duration_s))

    return -1

if __name__ == '__main__':
    print('Begin Genetic Algorithm Analysis')
    main(fname_config0)
    print('GA Completed')