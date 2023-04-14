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

from makeSim_nmatrix import createMatrix, createMatrix2
from GA_algorithm import read_from_config
from makeDepthScan import depth_scan_from_hdf, depth_scan_from_txt
from initialize import create_profile
from pleiades_scripting import make_command, test_job, submit_job, test_job_data, make_command_data, make_job, countjobs
from GA_selection import selection
sys.path.append('../')
import util
from data import bscan_rxList
from scipy.signal import correlate

if len(sys.argv) == 2:
    fname_config = sys.argv[1]
    fname_pseudo_external_str = 'None'
    fname_nmatrix_external_str = 'None'
    test_mode_str = 'False'
elif len(sys.argv) == 3:
    fname_config = sys.argv[1]
    fname_pseudo_external_str = sys.argv[2]
    fname_nmatrix_external_str = 'None'
    test_mode_str = False
elif len(sys.argv) == 4:
    fname_config = sys.argv[1]
    fname_pseudo_external_str = sys.argv[2]
    fname_nmatrix_external_str = sys.argv[3]
    test_mode_str = 'False'
elif len(sys.argv) == 5:
    fname_config = sys.argv[1]
    fname_pseudo_external_str = sys.argv[2]
    fname_nmatrix_external_str = sys.argv[3]
    test_mode_str = sys.argv[4]
else:
    print('error! wrong arg number:', len(sys.argv))
    sys.exit()

if test_mode_str == 'True':
    test_mode = True
elif test_mode_str == 'False':
    test_mode = False
else:
    test_mode = False

#Check if PseudoData Exists
if fname_pseudo_external_str == 'None':
    fname_pseudo_external = None
else:
    if os.path.isfile(fname_pseudo_external_str) == False:
        fname_pseudo_external = None
        print('Warning, Pseudodata File:', fname_pseudo_external_str, 'does not exist -> creating new one')
    else:
        fname_pseudo_external = fname_pseudo_external_str

#Check if nMatrix Exists
if fname_nmatrix_external_str == 'None':
    fname_nmatrix_external = None
else:
    if os.path.isfile(fname_nmatrix_external_str) == False:
        fname_nmatrix_external = None
        print('Warning, nmatrix file', fname_nmatrix_external_str, 'does not exist -> creating new one')
        print('nmatrix attempeted:', fname_nmatrix_external_str)
        sys.exit()
    else:
        print('Using old nmatrix')
        fname_nmatrix_external = fname_nmatrix_external_str

def main(fname_config, fname_pseudo_external = None, fname_nmatrix_external = None, test_mode=False):
    #Initialization
    print('start')
    start = time.time()

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

    # Save config file
    # config_cp = fname_config[:-4] + '_' + time_str + '.txt'
    config_cp = results_dir + '/config-file.txt'
    os.system('cp ' + fname_config + ' ' + config_cp)
    print('saving config file to ', config_cp)
    fname_pseudo_txt = config['INPUT']['fname_pseudodata']
    # Load Genetic Algorithm Properties
    print('load GA parameters')
    GA_1 = read_from_config(fname_config=config_cp)

    if fname_pseudo_external == None:
        fname_pseudo_output0 = config['OUTPUT']['fname_pseudo_output']
        fname_pseudo_output = fname_pseudo_output0[:-3] + '_' + time_str + '.h5'
    else:
        print('Old PsuedoData')
        fname_pseudo_output = fname_pseudo_external
    if fname_nmatrix_external == None:
        fname_nmatrix_output0 = config['OUTPUT']['fname_nmatrix_output']
        fname_nmatrix_output = fname_nmatrix_output0[:-3] + '_' + time_str + '.h5'
        ii_gen_complete = 0
        # Create Memmaps:
        fname_nmatrix_output_npy = fname_nmatrix_output[:-3] + '.npy'
        fname_nmatrix_output_misfit_npy = fname_nmatrix_output[:-3] + '_misfit.npy'

        util.create_memmap(fname_nmatrix_output_npy, dimensions=(GA_1.nGenerations, GA_1.nIndividuals),
                           data_type='float')
    else:
        print('Old Nmatrix')
        fname_nmatrix_output = fname_nmatrix_external
        fname_nmatrix_output_npy = fname_nmatrix_output[:-3] + '.npy'
        fname_nmatrix_output_misfit_npy = fname_nmatrix_output[:-3] + '_misfit.npy'

        nmatrix_npy = np.load(fname_nmatrix_output_npy,'r')
        misfit_npy = np.load(fname_nmatrix_output_misfit_npy, 'r')
        ii_gen_complete = 0
        for i in range(len(nmatrix_npy)):
            if np.all(nmatrix_npy[i] == 0) == False:
                ii_gen_complete += 1
        if ii_gen_complete > 0:
            hdf_nmatrix = h5py.File(fname_nmatrix_output, 'r+')
            S_arr = hdf_nmatrix['S_arr']
            misfit_arr = hdf_nmatrix['misfit_arr']
            for i in range(0, ii_gen_complete):
                if np.all(S_arr[i] == 0) == True and np.all(nmatrix_npy[i] == 0) == False:
                    S_arr[i] = nmatrix_npy[i]
                    misfit_arr[i] = misfit_npy[i]
            hdf_nmatrix.close()

    #PSEUDO-DATA
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

    nprofile_pseudo_genes = util.get_profile_from_file_decimate(fname=fname_pseudo_txt,
                                                                zmin=zMin_genes,
                                                                zmax=zMax_genes,
                                                                dz_out=dz_genes)
    #Make Pseudo-Data
    nprofile_pseudo = create_profile(zprof_out=zspace_simul, nprof_genes=nprofile_pseudo_genes,
                                     zprof_genes=zspace_genes,
                                     nprof_override=nprof_override,
                                     zprof_override=zprof_override)
    fname_nprofile_pseudo_txt = results_dir+'/nprofile_pseudo.txt'
    util.save_profile_to_txtfile(zprof=zspace_simul, nprof=nprofile_pseudo, fname=fname_nprofile_pseudo_txt)
    if ii_gen_complete == 0:
        if fname_pseudo_external == None:
            print('Make PsuedoData')
            depth_scan_from_txt(fname_config, fname_nprofile=fname_nprofile_pseudo_txt, fname_out=fname_pseudo_output)
        else:
            print('PseudoData already exists -> proceed')

    bscan_pseudo = bscan_rxList()
    bscan_pseudo.load_sim(fname_pseudo_output)
    nTX = bscan_pseudo.nTX
    nRX = bscan_pseudo.nRX
    util.create_memmap(fname_nmatrix_output_misfit_npy, dimensions=(GA_1.nGenerations, GA_1.nGenerations, nTX, nRX))

    #INITIALIZATION
    #TODO: Complete this... this is where the fun begins ;)
    #TODO: Add noise and proceed with GA
    #TODO: Add initial peak-based fits and proceed with GA

    #Generate Noise
    print('Generate ref-index noise -> create initial generation')
    nSamples = len(zspace_simul)
    genes_start = np.ones((GA_1.nIndividuals, GA_1.nGenes))
    nprofile_start = np.ones((GA_1.nIndividuals, nSamples))
    dn_offset = 0.02

    if test_mode == True:
        for i in range(GA_1.nIndividuals):

            n_noise = np.random.normal(0, dn_offset, GA_1.nGenes)
            genes_start[i] = nprofile_pseudo_genes + n_noise

            nprofile_start[i] = create_profile(zprof_out=zspace_simul, nprof_genes=genes_start[i],
                                               zprof_genes=zspace_genes,
                                               nprof_override=nprof_override,
                                               zprof_override=zprof_override)
    #TODO: Interpolate from TX -> RX direct peak guess -> genes -> Spline?
    '''
    else:
        bscan_pseudo = bscan_rxList()
        bscan_pseudo.load_sim(fname_pseudo_output)
        nTX = bscan_pseudo.nTX
        nRX = bscan_pseudo.nRX
        tx_depths = bscan_pseudo.tx_depths
        rxList = bscan_pseudo.rxList

        #Select Peaks
        rx_id_list = []

        #RX Ranges From Config File:
        fname_rx = config['RECEIVER']['fname_receivers']
        rx_arr = np.genfromtxt(fname_rx, skip_header=1)
        rxRanges = np.unique(rx_arr[:,0])

        for k in range(len(rxRanges)):
            x_k = rxRanges[k]
            rx_id = []
            for i in range(nTX):
                for j in range(nRX):
                    z_tx = tx_depths[i]
                    z_rx = rxList[j].z
                    if z_tx == z_rx and rxList[j].x == x_k:
                        rx_id.append(j)
            rx_id_list.append(rx_id)
        bscan_pseudo_npy = bscan_pseudo.bscan_sig
        tx_signal = bscan_pseudo.tx_signal
        tspace = tx_signal.tspace
        tmax = max(tspace)

        nprof_guess_list = []
        for k in range(len(rxRanges)):
            t_peaks_max = np.ones(nTX)
            t_peaks_first = np.ones(nTX)
            R = rxRanges[k]
            for i in range(nTX):
                j = rx_id_list[k][i]
                sig_pseudodata = bscan_pseudo_npy[i,j]
                sig_correl = abs(correlate(sig_pseudodata, tx_signal.pulse))
                nLag = len(sig_correl)
                t_lag = np.linspace(-tmax, tmax, nLag)
                j_cut = util.findNearest(tspace, 0)
                tspace_cut = t_lag[j_cut:]
                sig_correl_cut = sig_correl[j_cut:]

                inds = pku.indexes(sig_correl_cut)
                t_peaks_first[i] = tspace_cut[inds[0]]
                j_max = np.argmax(sig_correl_cut)
                t_peaks_max[i] = tspace_cut[j_max]
            c0 = 0.3
            nprof_first = c0*t_peaks_first/R
            nprof_max = c0*t_peaks_max/R
            nprof_guess_list.append(nprof_first)
            nprof_guess_list.append(nprof_max)
        
    '''
    nStart = 1000
    gene_pool = np.ones((nStart, GA_1.nGenes))
    for i in range(nStart):
        n_noise = np.random.normal(0, dn_offset, GA_1.nGenes)
        gene_pool[i] = nprofile_pseudo_genes + n_noise
    if ii_gen_complete == 0:
        createMatrix2(fname_config=fname_config, n_prof_initial=nprofile_start, genes_initial=genes_start,
                      z_profile= zspace_simul, z_genes=zspace_genes, fname_nmatrix=fname_nmatrix_output,
                      nGenerations=GA_1.nGenerations)

    cmd_prefix = 'python runSimulation.py '
    ii_gen = 0

    dir_outfiles0 = results_dir + '/outfiles'
    os.system('mkdir ' + dir_outfiles0)
    #tstart_1st_gen = time.time()
    if ii_gen_complete == 0:
        print('Initialize, create first generation')
        for jj_ind in range(GA_1.nIndividuals):
            print('Individual', jj_ind)
            cmd_j = cmd_prefix + ' ' + fname_config + ' ' + fname_pseudo_output + ' ' + fname_nmatrix_output + ' ' + str(ii_gen) + ' ' + str(jj_ind)
            if test_mode == True:
                os.system(cmd_j)
            else:
                dir_outfiles = dir_outfiles0 + '/' + 'gen' + str(ii_gen)
                jobname = job_prefix + str(ii_gen) + '-' + str(jj_ind)
                sh_file = jobname + '.sh'
                out_file = dir_outfiles + '/' + jobname + '.out'
                #outfile_list.append(out_file)
                make_job(sh_file, out_file, jobname, cmd_j)
                submit_job(sh_file)
                os.system('rm -f ' + sh_file)

        if test_mode == False:
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
    #tend_1st_gen = time.time()
    #duration_1st_gen = tend_1st_gen - tstart_1st_gen

    # Wait for jobs to be submitted
    print('1st generation finished')
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
    fout.write(fname_pseudo_output + '\t' + fname_nmatrix_output + '\n')
    fout.write('gen\tind\tS\tfname_out\n')
    fout.close()
    tsleep = 10.
    #max_time = 2 * duration_1st_gen
    t_cycle = 0

    for ii_gen in range(ii_gen_complete, GA_1.nGenerations):
        #nJobs = countjobs()
        nJobs = 0
        #TODO -> FIx
        print('Generation', ii_gen, 'Check jobs')

        #Check If Any Jobs still running, Cancel if So
        '''
        if (nJobs == 0) or (t_cycle > max_time):
            print('Submit Jobs Now \n')
            if t_cycle > max_time and nJobs > 0:
                kk = 1
                nJobs_2 = countjobs()
                while nJobs_2 > 0 and kk <= 10:
                    print('Processes still running, N = ', nJobs)
                    print('Running kill command')
                    os.system('python kill-jobs.py')
                    print('Wait')
                    time.sleep(tsleep)
                    nJobs_2 = countjobs()
                    kk += 1
                    print(nJobs_2)
        '''
        if nJobs == 0:
            # Save Scores from Last Generation from NPY to HDF File
            S_arr_npy = np.load(fname_nmatrix_output_npy, 'r')
            misfit_matrix_npy = np.load(fname_nmatrix_output_misfit_npy, 'r')
            print('True or False?', np.all(S_arr_npy == 0))
            #TODO: Why am I selecting the wrong fitness values?????
            print(misfit_matrix_npy[ii_gen - 1])
            print('S_list, gen = ', ii_gen - 1, '\n', S_arr_npy[ii_gen - 1])
            nmatrix_hdf = h5py.File(fname_nmatrix_output, 'r+')
            S_arr_last = nmatrix_hdf['S_arr']
            misfit_arr = nmatrix_hdf['misfit_arr']
            print('ii_gen=', ii_gen, S_arr_last, misfit_arr)
            print('Set S')
            print(S_arr_npy[ii_gen-1], S_arr_npy[ii_gen])
            S_arr_last[ii_gen - 1] = S_arr_npy[ii_gen - 1]

            print('Set Misfit')
            #print(misfit_arr[ii_gen-1,0,0,0])
            #TODO: See why this gives me value error
            #misfit_arr[ii_gen - 1, 0, 0, 0] = misfit_matrix_npy[ii_gen - 1, 0, 0,0]
            nmatrix_hdf.close()

            # Apply GA Selection
            print('Applying Selection Routines')
            nmatrix_hdf = h5py.File(fname_nmatrix_output, 'r+')
            S_arr = np.array(nmatrix_hdf['S_arr'])
            n_profile_matrix = nmatrix_hdf['n_profile_matrix']
            genes_matrix = nmatrix_hdf['genes_matrix']
            nprof_parents = genes_matrix[ii_gen - 1]
            S_list = np.array(S_arr[ii_gen - 1])
            S_max = max(S_list)

            # Select Genes
            print(S_list)
            n_profile_children_genes = selection(prof_list=nprof_parents, S_list=S_list,
                                                 prof_list_initial=gene_pool,
                                                 f_roulette=GA_1.fRoulette, f_elite=GA_1.fElite,
                                                 f_cross_over=GA_1.fCrossOver, f_immigrant=GA_1.fImmigrant,
                                                 f_mutant=GA_1.fMutation, mutation_thres=mutation_thres)
            # Create New Ref-Index Profiles
            for j in range(GA_1.nIndividuals):
                nprof_children_genes_j = n_profile_children_genes[j]  # TODO: Check that x-y size is equal
                nprof_children_j = create_profile(zspace_simul, nprof_genes=nprof_children_genes_j,
                                                  zprof_genes=zspace_genes,
                                                  nprof_override=nprof_override,
                                                  zprof_override=zprof_override)
                n_profile_matrix[ii_gen, j] = nprof_children_j
                genes_matrix[ii_gen, j] = nprof_children_genes_j
            nmatrix_hdf.close()
            t_cycle = 0
            jj_best = np.argmax(S_arr)
            S_max_list.append(S_max)
            S_mean = np.mean(S_list)
            S_var = np.std(S_list)
            S_med = np.median(S_list)

            S_mean_list.append(S_mean)
            S_var_list.append(S_var)
            S_med_list.append(S_med)
            nmatrix_hdf.close()

            gens = np.arange(0, ii_gen, 1)

            # Make Plots:
            fig = pl.figure(figsize=(8, 5), dpi=120)
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
            f_log = open(fname_log, 'a')
            f_log.write(line)
            f_log.close()

            # Submit Jobs:
            print('Run Jobs:')
            for j in range(GA_1.nIndividuals):
                print('Individual ', j)
                cmd_j = cmd_prefix + ' ' + config_cp + ' ' + fname_pseudo_output + ' ' + fname_nmatrix_output + ' ' + str(
                    ii_gen) + ' ' + str(j)
                print(test_mode)
                if test_mode == False:
                    print('Submit to Cluster')
                    # Create Command
                    dir_outfiles = dir_outfiles0 + '/' + 'gen' + str(ii_gen)
                    if os.path.isdir(dir_outfiles) == False:
                        os.system('mkdir ' + dir_outfiles)

                    jobname = job_prefix + str(ii_gen) + '-' + str(j)
                    sh_file = jobname + '.sh'
                    out_file = dir_outfiles + '/' + jobname + '.out'
                    # outfile_list.append(out_file)
                    make_job(sh_file, out_file, jobname, cmd_j)
                    submit_job(sh_file)
                    # After Jobs are submitted
                    os.system('rm -f ' + sh_file)
                else:
                    # TODO: Resource Unavailable Error!! FFS -> Start from Scatch
                    print('Run Directly')
                    os.system(cmd_j)
            print('Jobs running -> Generation: ', ii_gen)

            fname_pseudo_output2 = results_dir + '/' + fname_pseudo_output
            fname_nmatrix_output2 = results_dir + '/' + fname_nmatrix_output
            fname_nmatrix_output2_npy = fname_nmatrix_output2[:-3] + '.npy'
            os.system('cp ' + fname_pseudo_output + ' ' + fname_pseudo_output2)
            os.system('cp ' + fname_nmatrix_output + ' ' + fname_nmatrix_output2)
            os.system('cp ' + fname_nmatrix_output_npy + ' ' + fname_nmatrix_output2_npy)
        else:
            print('Queue of jobs: ', nJobs)
            print('Wait:', tsleep, ' seconds')
            t_cycle += tsleep
            print('Elapsed seconds: ', t_cycle)
            print('Elapsed time: ', datetime.timedelta(seconds=t_cycle))
            print('')
            time.sleep(tsleep)



if __name__ == '__main__':
    print('Run GA')
    main(fname_config=fname_config, fname_pseudo_external=fname_pseudo_external,
         fname_nmatrix_external=fname_nmatrix_external, test_mode=test_mode)
    print('End GA')