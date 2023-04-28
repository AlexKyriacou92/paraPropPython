import datetime
import os
import random
import sys
from math import pi
import peakutils as pku
import subprocess
import time
import configparser

import h5py
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as pl
import sys
import random

from makeSim_nmatrix import createMatrix, createMatrix2
from GA_algorithm import read_from_config
from makeDepthScan import depth_scan_from_hdf, depth_scan_from_txt
from initialize import create_profile, initialize
from pleiades_scripting import make_command, test_job, submit_job, test_job_data, make_command_data, make_job, countjobs
from GA_selection import selection
sys.path.append('../')
import util
from data import bscan_rxList
from scipy.signal import correlate

import matplotlib.pyplot as pl


def main(fname_config, fname_data, fname_nmatrix_external = None, test_mode = False, parallel_mode = True):
    #Initialization
    print('start')
    #Load Simulation Config
    print('load simulation config')
    config = configparser.ConfigParser()
    config.read(fname_config)

    #Datetime
    now = datetime.datetime.now()
    time_str = now.strftime('%y%m%d_%H%M%S')

    # Set up Simulation Output Files
    print('set up output files')
    sim_mode = config['INPUT']['sim_mode']
    job_prefix = config['INPUT']['prefix']

     # Create directory to store results later
    results_dir = job_prefix + '_' + time_str
    os.system('mkdir ' + results_dir)


    fname_pseudo_txt = config['INPUT']['fname_pseudodata']
    # Load Genetic Algorithm Properties
    print('load GA parameters')
    GA_1 = read_from_config(fname_config=fname_config)
    if fname_nmatrix_external == None:
         fname_nmatrix_output0 = config['OUTPUT']['fname_nmatrix_output']
         fname_nmatrix_output = fname_nmatrix_output0[:-3] + '_' + time_str + '.h5'
         ii_gen_complete = 0
         # Create Memmaps:
         fname_nmatrix_output_npy = fname_nmatrix_output[:-3] + '.npy'
         # fname_nmatrix_output_misfit_npy = fname_nmatrix_output[:-3] + '_misfit.npy'
         util.create_memmap(fname_nmatrix_output_npy, dimensions=(GA_1.nGenerations, GA_1.nIndividuals), data_type='float')
    else:
        print('Old Nmatrix')
        fname_nmatrix_output = fname_nmatrix_external
        fname_nmatrix_output_npy = fname_nmatrix_output[:-3] + '.npy'
        nmatrix_npy = np.load(fname_nmatrix_output_npy,'r')
        ii_gen_complete = 0
        for i in range(len(nmatrix_npy)):
            if np.all(nmatrix_npy[i] == 0) == False:
                ii_gen_complete += 1
        if ii_gen_complete > 0:
            nmatrix_hdf = h5py.File(fname_nmatrix_output, 'r+')
            S_arr = nmatrix_hdf['S_arr']
            #misfit_arr = nmatrix_hdf['misfit_arr']
            for i in range(0, ii_gen_complete):
                if np.all(S_arr[i] == 0) == True and np.all(nmatrix_npy[i] == 0) == False:
                    S_arr[i] = nmatrix_npy[i]
                    #misfit_arr[i] = misfit_npy[i]
            nmatrix_hdf.close()

    fname_override = config['Override']['fname_override']
    nprof_override, zprof_override = util.get_profile_from_file(fname_override)

    mutation_thres = float(config['GA']['mutation_thres'])

    zMin_genes = float(config['GA']['minDepth'])
    zMax_genes = float(config['GA']['maxDepth'])
    iceDepth = float(config['GEOMETRY']['iceDepth'])
    dz = float(config['GEOMETRY']['dz'])
    print('Mutation thresold:', mutation_thres)
    print('zMin genes:', zMin_genes, 'zMax genes', zMax_genes, 'dz:', dz)

    # Set Overide
    zMin_genes = float(config['GA']['minDepth'])
    zMax_genes = float(config['GA']['maxDepth'])
    iceDepth = float(config['GEOMETRY']['iceDepth'])
    dz = float(config['GEOMETRY']['dz'])
    zspace_simul = np.arange(0, iceDepth + dz, dz)
    zspace_genes = np.linspace(zMin_genes, zMax_genes, GA_1.nGenes)
    dz_genes = zspace_genes[1] - zspace_genes[0]

    #====================================
    hdf_data = h5py.File(fname_data, 'r')
    fftArray = np.array(hdf_data['fftArray'])
    freqList = np.array(hdf_data['freqList']) / 1e9 #Note FT Data is defined in s/Hz, while paraProp uses ns/GHz
    rxDepths = np.array(hdf_data['rxDepths'])
    rxRanges = np.array(hdf_data['rxRanges'])
    tspace_data = np.array(hdf_data['tspace']) * 1e9 #Note FT Data is defined in s/Hz, while paraProp uses ns/GHz
    #nSamples = len(tspace_data)
    txDepths = np.array(hdf_data['txDepths'])
    hdf_data.close()
    nMeasurements = len(fftArray)

    ztx_depths_unique = np.unique(txDepths)
    R_unique = np.unique(rxRanges)
    nTx_unique = len(ztx_depths_unique)

    nprof_gene_guess = []
    zprof_gene_guess = []
    for k in range(len(R_unique)):
        R = R_unique[k]
        tx_depth_list = []
        rx_depth_list = []
        fftArray_list = []
        #Select Parallel Depths
        nTX2 = 0
        zprof = []
        for i in range(nTx_unique):
            if ztx_depths_unique[i] >= max(zprof_override):
                for j in range(nMeasurements):
                    if ztx_depths_unique[i] == txDepths[j] and ztx_depths_unique[i] == rxDepths[j] and rxRanges[j] == R_unique[k]:
                        tx_depth_list.append(txDepths[j])
                        rx_depth_list.append(rxDepths[j])
                        fftArray_list.append(abs(fftArray[j])**2)
                        zprof.append(ztx_depths_unique[i])
                        nTX2 += 1
        zprof = np.array(zprof)
        nprof_first = np.ones(nTX2)
        nprof_max = np.ones(nTX2)
        for i in range(nTx_unique):
            fft_i = fftArray_list[i]
            jj_pks = pku.indexes(fft_i, thres=0.1)
            nPeaks = len(jj_pks)
            fft_peaks = []
            t_peaks = []
            for j in range(nPeaks):
                jj = jj_pks[j]
                fft_peaks.append(fft_i[jj])
                t_peaks.append(tspace_data[jj])
            t_peak_first = t_peaks[0]
            k_max = np.armgax(fft_peaks)
            t_peak_max = t_peaks[k_max]
            n_first = 0.3 * t_peak_first / R
            n_max = 0.3 * t_peak_max / R
            nprof_first[i] = n_first
            nprof_max[i] = n_max
        zprof_gene_guess.append(zprof)
        zprof_gene_guess.append(zprof)
        nprof_gene_guess.append(nprof_first)
        nprof_gene_guess.append(nprof_max)

    nGuess = len(nprof_gene_guess)
    genes_from_peak_list = []
    for i in range(nGuess):
         nprof_i = nprof_gene_guess[i]
         zprof_i = zprof_gene_guess[i]
         if max(zspace_genes) > max(zprof_i):
             zprof_i = np.append(zprof_i, max(zspace_genes))
             nprof_i = np.append(nprof_i, np.random.uniform(1.2, 1.7, 1))
         nprof_genes, zprof_genes = util.do_interpolation_same_depth(zprof_in=zprof_i, nprof_in=nprof_i,
                                                                     N=GA_1.nGenes)
         genes_from_peak_list.append(nprof_genes)

    nDepths = len(zspace_simul)
    genes_start = np.ones((GA_1.nIndividuals, GA_1.nGenes))
    nprofile_start = np.ones((GA_1.nIndividuals, nDepths))

    nStart = 10000
    nQuarter = nStart // nGuess
    gene_pool = []

    fAnalytical = float(config['GA']['fAnalytical'])
    fFluctuations = float(config['GA']['fFluctuations'])
    fFlat = float(config['GA']['fFlat'])
    fSine = float(config['GA']['fSine'])
    fExp = float(config['GA']['fExp'])
    for i in range(nGuess):
        genes_start[i] = genes_from_peak_list[i]
        nprofile_start[i] = create_profile(zspace_simul, nprof_genes=genes_start[i], zprof_genes=zspace_genes,
                                           nprof_override=nprof_override, zprof_override=zprof_override)
    for i in range(nGuess):
        genes_i = initialize(nStart=nQuarter, nprofile_sampling_mean=genes_from_peak_list[i],
                             zprofile_sampling_mean=zspace_genes,
                             GA=GA_1, fAnalytical=fAnalytical, fFluctuations=fFluctuations,
                             fFlat=fFlat, fSine=fSine, fExp=fExp)
        for j in range(nQuarter):
            gene_pool.append(genes_i[j])

    for i in range(nGuess, GA_1.nIndividuals):
        genes_rand = random.sample(gene_pool, 1)
        genes_start[i] = genes_rand[0]
        nprofile_start[i] = create_profile(zspace_simul, nprof_genes=genes_start[i], zprof_genes=zspace_genes,
                                           nprof_override=nprof_override, zprof_override=zprof_override)
    #Inititialization Complete

    if ii_gen_complete == 0:
        print('Create nmatrix')
        createMatrix2(fname_config=fname_config, n_prof_initial=nprofile_start, genes_initial=genes_start,
                      z_profile= zspace_simul, z_genes=zspace_genes, fname_nmatrix=fname_nmatrix_output,
                      nGenerations=GA_1.nGenerations)
    else:
        print('Skip creating nmatrix -> already exists!')

    cmd_prefix = 'python runSimulation_Data.py '
    dir_outfiles0 = results_dir + '/outfiles'
    os.system('mkdir ' + dir_outfiles0)
    #tstart_1st_gen = time.time()
    # Save config file
    # DO THIS LATER
    config_cp = results_dir + '/config-file.txt'
    os.system('cp ' + fname_config + ' ' + config_cp)
    print('saving config file to ', config_cp)
    if ii_gen_complete == 0:
        print('Initialize, create first generation')
        outfile_list = []
        ii_gen = ii_gen_complete

        for jj_ind in range(GA_1.nIndividuals):
            print('Individual', jj_ind)
            cmd_j = cmd_prefix + ' ' + fname_config + ' ' + fname_data + ' ' + fname_nmatrix_output + ' ' + str(ii_gen) + ' ' + str(jj_ind)
            if parallel_mode == False:
                os.system(cmd_j)
            else:
                dir_outfiles = dir_outfiles0 + '/' + 'gen' + str(ii_gen)
                if os.path.isdir(dir_outfiles) == False:
                    os.system('mkdir ' + dir_outfiles)
                jobname = job_prefix + str(ii_gen) + '-' + str(jj_ind)
                sh_file = jobname + '.sh'
                out_file = dir_outfiles + '/' + jobname + '.out'
                outfile_list.append(out_file)
                make_job(sh_file, out_file, jobname, cmd_j)
                submit_job(sh_file)
                os.system('rm -f ' + sh_file)
        if parallel_mode == True:
            print('jobs submitted')
            proceed_bool = False
            while proceed_bool == False:
                tsleep = 10.
                nJobs = countjobs()
                print(nJobs, 'jobs left, wait:', tsleep)
                if nJobs > 0:
                    time.sleep(tsleep)
                else:
                    proceed_bool = True

        ii_gen_complete += 1
    else:
        print(ii_gen_complete, ' generations already finished, proceed->')

    ii_gen = ii_gen_complete

    # Wait for jobs to be submitted
    print('next generation:')
    
    S_max = 0
    S_max_list = []
    S_mean_list = []
    S_var_list = []
    S_med_list = []

    fname_log = results_dir + '/log_report.txt'
    # f_log = open(fname_log,'w')
    # f_log.write('gen\tS_max\tS_mean\tS_var\tS_med\n')
    f_log = open(fname_log, 'w')
    f_log.write('gen\tS_max\tS_mean\tS_var\tS_med\n')
    f_log.close()

    fname_report = results_dir + '/' + 'simul_report.txt'
    fout = open(fname_report, 'w')
    nOutput = GA_1.nGenerations % 10 + 4
    fout.write('pathto\tnOutput \n' + results_dir + '\t' + str(nOutput) + '\n' + '\n')
    fout.write('fname_psuedo_data\tfname_nmatrix\n')
    fout.write(fname_data + '\t' + fname_nmatrix_output + '\n')
    fout.write('gen\tind\tS\tfname_out\n')
    fout.close()
    '''
    matrix_hdf = h5py.File(fname_nmatrix_output, 'r')
    duration_sim = float(nmatrix_hdf['duration'])
    nmatrix_hdf.close()
    '''
    duration_sim = 10 * 60.
    tsleep = 10.
    max_time = 4 * duration_sim
    t_cycle = 0
    
    print('Starting Generation Scan:, gen:', ii_gen)
    while ii_gen < GA_1.nGenerations:
        nJobs = countjobs()
        print('Check jobs')
        print('')
        if nJobs == 0:
            print('Generation: ', ii_gen-1, ' complete')

            print('Save scores')
            # Save Scores from Last Generation from NPY to HDF File
            S_arr_npy = np.load(fname_nmatrix_output_npy, 'r')
            #misfit_matrix_npy = np.load(fname_nmatrix_output_misfit_npy, 'r')
            S_list = S_arr_npy[ii_gen-1]

            print('S_list, gen = ', ii_gen - 1, '\n', S_list)
            nmatrix_hdf = h5py.File(fname_nmatrix_output, 'r+')
            S_arr = nmatrix_hdf['S_arr']
            #misfit_arr = nmatrix_hdf['misfit_arr']
            print('Set S')
            S_arr[ii_gen - 1] = S_list

            #print(misfit_arr[ii_gen-1,0,0,0])
            #misfit_arr[ii_gen - 1, 0, 0, 0] = misfit_matrix_npy[ii_gen - 1, 0, 0,0]

            n_profile_matrix = nmatrix_hdf['n_profile_matrix']
            genes_matrix = nmatrix_hdf['genes_matrix']
            nprof_parents = genes_matrix[ii_gen - 1]
            S_max = max(S_list)
            jj_best = np.argmax(S_list)
            t_cycle = 0
            S_max_list.append(S_max)
            S_mean = np.mean(S_list)
            S_var = np.std(S_list)
            S_med = np.median(S_list)

            S_mean_list.append(S_mean)
            S_var_list.append(S_var)
            S_med_list.append(S_med)

            nprof_best = nprof_parents[jj_best]

            print('Highest Score from gen:', ii_gen-1, ', S_max=', S_max, 'ind:', jj_best)
            print('Lowest misfit: ', ii_gen-1, ' misfit_min=', 1/S_max)
            print('Median Score S_median=', S_med)
            if ii_gen > 2:
                S_ratio = S_max/max(S_arr[ii_gen-2])
                S_ratio2 = S_med/np.median(S_arr[ii_gen-2])
                print('Change of max score from previous:', S_ratio)
                print('Change of median score from previous:', S_ratio2)
            # Select Genes
            print('Selecting Genes')
            n_profile_children_genes = selection(prof_list=nprof_parents, S_list=S_list,
                                                 prof_list_initial=gene_pool,
                                                 f_roulette=GA_1.fRoulette, f_elite=GA_1.fElite,
                                                 f_cross_over=GA_1.fCrossOver, f_immigrant=GA_1.fImmigrant,
                                                 f_mutant=GA_1.fMutation, mutation_thres=mutation_thres)
            print('Selection complete')
            # Create New Ref-Index Profiles
            print('Reproducing')
            for j in range(GA_1.nIndividuals):
                nprof_children_genes_j = n_profile_children_genes[j]  # TODO: Check that x-y size is equal
                nprof_children_j = create_profile(zspace_simul, nprof_genes=nprof_children_genes_j,
                                                  zprof_genes=zspace_genes,
                                                  nprof_override=nprof_override,
                                                  zprof_override=zprof_override)
                n_profile_matrix[ii_gen, j] = nprof_children_j
                genes_matrix[ii_gen, j] = nprof_children_genes_j
            nmatrix_hdf.close()
            print('Closing nmatrix hdf')

            print('Writing to log')
            line = str(ii_gen) + '\t' + str(S_max) + '\t' + str(S_mean) + '\t' + str(S_var) + '\t' + str(S_med) + '\n'
            f_log = open(fname_log, 'a')
            f_log.write(line)
            f_log.close()

            # Submit Jobs:
            print('Run Jobs:')
            for j in range(GA_1.nIndividuals):
                print('Individual ', j)
                cmd_j = cmd_prefix + ' ' + config_cp + ' ' + fname_data + ' ' + fname_nmatrix_output + ' ' + str(
                    ii_gen) + ' ' + str(j)
                print(parallel_mode)
                if parallel_mode == True:
                    print('Submit to Cluster')
                    outfile_list = []
                    # Create Command
                    dir_outfiles = dir_outfiles0 + '/' + 'gen' + str(ii_gen)
                    if os.path.isdir(dir_outfiles) == False:
                        os.system('mkdir ' + dir_outfiles)

                    jobname = job_prefix + str(ii_gen) + '-' + str(j)
                    sh_file = jobname + '.sh'
                    out_file = dir_outfiles + '/' + jobname + '.out'
                    outfile_list.append(out_file)
                    make_job(sh_file, out_file, jobname, cmd_j)
                    submit_job(sh_file)
                    # After Jobs are submitted
                    os.system('rm -f ' + sh_file)
                    time.sleep(0.5)
                else:
                    print('Run Directly')
                    os.system(cmd_j)
                    time.sleep(0.5)
            if parallel_mode == True:
                print('All jobs submitted -> Generation: ', ii_gen)

            fname_data2 = results_dir + '/' + fname_data
            fname_nmatrix_output2 = results_dir + '/' + fname_nmatrix_output
            fname_nmatrix_output2_npy = fname_nmatrix_output2[:-3] + '.npy'
            print('Copying all data to dir: ', results_dir)
            os.system('cp ' + fname_data + ' ' + fname_data2)
            os.system('cp ' + fname_nmatrix_output + ' ' + fname_nmatrix_output2)
            os.system('cp ' + fname_nmatrix_output_npy + ' ' + fname_nmatrix_output2_npy)
            ii_gen += 1
            print('')
        else:
            print('Queue of jobs: ', nJobs)
            print('Wait:', tsleep, ' seconds')
            t_cycle += tsleep
            print('Elapsed seconds: ', t_cycle)
            print('Elapsed time: ', datetime.timedelta(seconds=t_cycle))
            print('')
            time.sleep(tsleep)
    
    #============================================================================================================
    return -1

if len(sys.argv) == 3:
    fname_config = sys.argv[1]
    fname_data = sys.argv[2]
    fname_nmatrix_external_str = 'None'
    test_mode_str = 'False'
    parallel_mode_str = 'True'
elif len(sys.argv) == 4:
    fname_config = sys.argv[1]
    fname_data = sys.argv[2]
    fname_nmatrix_external_str = sys.argv[3]
    test_mode_str = 'False'
    parallel_mode_str = 'True'
elif len(sys.argv) == 5:
    fname_config = sys.argv[1]
    fname_data = sys.argv[2]
    fname_nmatrix_external_str = sys.argv[3]
    test_mode_str = sys.argv[4]
    parallel_mode_str = 'True'
elif len(sys.argv) == 6:
    fname_config = sys.argv[1]
    fname_data = sys.argv[2]
    fname_nmatrix_external_str = sys.argv[3]
    test_mode_str = sys.argv[4]
    parallel_mode_str = sys.argv[5]
else:
    print('error! wrong arg number:', len(sys.argv))
    sys.exit()

    sys.exit()

if test_mode_str == 'True':
    test_mode = True
elif test_mode_str == 'False':
    test_mode = False
else:
    test_mode = False

if parallel_mode_str == 'True':
    parallel_mode = True
elif parallel_mode_str == 'False':
    parallel_mode = False
else:
    parallel_mode = False

#Check if nMatrix Exists
if fname_nmatrix_external_str == 'None':
    fname_nmatrix_external = None
else:
    if os.path.isfile(fname_nmatrix_external_str) == False:
        fname_nmatrix_external = None
        print('Warning, nmatrix file', fname_nmatrix_external_str, 'does not exist -> creating new one')
        print('nmatrix attempeted:', fname_nmatrix_external_str)
        #sys.exit()
    else:
        print('Using old nmatrix')
        fname_nmatrix_external = fname_nmatrix_external_str


if __name__ == '__main__':
    print('Run GA')
    main(fname_config=fname_config, fname_data=fname_data,
         fname_nmatrix_external=fname_nmatrix_external,
         test_mode=test_mode, parallel_mode=parallel_mode)
    print('End GA')