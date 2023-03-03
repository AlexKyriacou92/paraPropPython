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
from makeSim_nmatrix import createMatrix
import sys

from genetic_functions import initialize_from_analytical, roulette, initalize_from_fluctuations, initialize
from pleiades_scripting import make_command, test_job, submit_job, test_job_data, make_command_data, make_job, countjobs
from selection_functions import selection

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

    #Selecting Initiate Populations
    print('initializing populations of ref-index profiles')
    n_prof_pool = initialize(nStart, nprof_sample_mean, zprof_sample_mean, GA_1, fAnalytical, fFluctuations,
                             fFlat, fSine, fExp)
    GA_1.initialize_from_sample(n_prof_pool)

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
    createMatrix(fname_config=config_cp, n_prof_initial=GA_1.first_generation, z_profile=zprof_sample_mean,
                 fname_nmatrix=fname_nmatrix, nGenerations=GA_1.nGenerations)
    hdf_nmatrix = h5py.File(fname_nmatrix, 'r+')
    hdf_nmatrix.attrs['datetime'] = time_str
    hdf_nmatrix.attrs['config_file'] = config_cp
    hdf_nmatrix.close()

    print('selecting simulation mode:', sim_mode)
    dir_outfiles0 = results_dir + '/outfiles'
    os.system('mkdir ' + dir_outfiles0)
    if sim_mode == 'pseudo':
        fname_pseudodata = config['INPUT']['fname_pseudodata']

        # Create Pseudo_Data Profile



        # Create Pseudo_Data Bscan
        print('create pseudo data')
        if os.path.isfile(fname_pseudo_output) == True:
            os.system('rm -f ' + fname_pseudo_output)
        cmd = 'python runSim_pseudodata_from_txt.py ' + config_cp + ' ' + fname_pseudodata + ' ' + fname_pseudo_output
        os.system(cmd)

    print('create n_matrix')

    return -1

if __name__ == '__main__':
    print('Begin Genetic Algorithm Analysis')
    main(fname_config0)
    print('GA Completed')