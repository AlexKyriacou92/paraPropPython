import datetime
import os
import random
import h5py
import numpy as np
from scipy.interpolate import interp1d
import subprocess
import time
from math import pi

from genetic_algorithm import GA, read_from_config
from makeSim_nmatrix import createMatrix
import sys
sys.path.append('../genetic_algorithm_analysis/')
from genetic_functions import initialize_from_analytical, roulette, initalize_from_fluctuations
from pleiades_scripting import make_command, test_job, submit_job, test_job_data, make_command_data, make_job
from selection_functions import selection


#=======================================================================================================================
# Initialize the GA
#=======================================================================================================================

def countjobs():
    cmd = 'squeue | grep "kyriacou" | wc -l'
    try:
        output = int(subprocess.check_output(cmd, shell=True))
    except:
        output = 0
    return output

def get_profile_from_file(fname):
    profile_data = np.genfromtxt(fname)
    z_profile = profile_data[:,0]
    n_profile = profile_data[:,1]
    return n_profile, z_profile

def do_interpolation(zprof_in, nprof_in, N):
    f_interp = interp1d(zprof_in, nprof_in)
    zprof_out = np.linspace(min(zprof_in),max(zprof_in), N)
    nprof_out = np.ones(N)
    nprof_out[0] = nprof_in[0]
    nprof_out[-1] = nprof_in[-1]
    nprof_out[1:-1] = f_interp(zprof_out[1:-1])
    return nprof_out, zprof_out

fname_config = 'config_aletsch_GA.txt'
#fname_nprof_psuedodata = 'start_profiles/aletsch_glacier_2.txt'
GA_1 = read_from_config(fname_config=fname_config)
print('nIndividuas:',GA_1.nIndividuals)

fname_nprofile_sampling_mean = 'start_profiles/aletsch_glacier_1.txt'
nprofile_sampling_mean_0, zprofile_sampling_mean_0 = get_profile_from_file(fname_nprofile_sampling_mean)
nprofile_sampling_mean, zprofile_sampling_mean = do_interpolation(zprofile_sampling_mean_0, nprofile_sampling_mean_0, GA_1.nGenes)

nStart = 10000 #Starting Sample

dz_start = zprofile_sampling_mean[1] - zprofile_sampling_mean[0]
nSamples_start = len(zprofile_sampling_mean)
print(dz_start, nSamples_start, zprofile_sampling_mean[0], zprofile_sampling_mean[-1])

n_prof_pool = []
nQuarter = nStart // 4
nprof_analytical = initialize_from_analytical(nprofile_sampling_mean, 0.04*np.ones(GA_1.nGenes), nQuarter)
nprof_flucations = initalize_from_fluctuations(nprofile_sampling_mean, zprofile_sampling_mean, nQuarter)
for i in range(nQuarter):
    n_prof_pool.append(nprof_analytical[i])
for i in range(nQuarter):
    n_prof_pool.append(nprof_flucations[i])
for i in range(nQuarter):
    n_const = random.uniform(0,0.78)
    n_prof_flat = np.ones(GA_1.nGenes) + n_const
    n_prof_pool.append(n_prof_flat)
for i in range(nQuarter):
    amp_rand = 0.4*random.random()
    z_period = random.uniform(0.5,15)
    k_factor = 1/z_period
    phase_rand = random.uniform(0, 2*pi)
    freq_rand = amp_rand*np.sin(2*pi*zprofile_sampling_mean*k_factor + phase_rand)
    n_prof_flat = np.ones(GA_1.nGenes) + n_const
    n_prof_pool.append(n_prof_flat)

random.shuffle(n_prof_pool)
GA_1.initialize_from_sample(n_prof_pool)
print(len(GA_1.first_generation))
print(type(GA_1.first_generation), GA_1.first_generation.shape)
fname_data = 'Field-Test-data.h5'

#Create nMatirx


#Create n_matrix

fname_nmatrix = 'test_nmatrix_data.h5' #Results will be saved here
createMatrix(fname_config=fname_config, n_prof_initial=GA_1.first_generation, z_profile=zprofile_sampling_mean,
             fname_nmatrix=fname_nmatrix, nGenerations = GA_1.nGenerations)
#First Population Set

#Next -> Calculate list of S parameters

ii_gen = 0 #Zeroth Generation

cmd_prefix = 'python runSim_nProfile_FT.py '

'''
nmatrix_hdf = h5py.File(fname_nmatrix, 'r+')
S_arr = np.array(nmatrix_hdf['S_arr'])
n_profile_matrix = np.array(nmatrix_hdf['n_profile_matrix'])
nmatrix_hdf.close()
print(n_profile_matrix[0])
'''

#Next -> Calculate list of S parameters
#Submit jobs to cluster and wait


for j in range(GA_1.nIndividuals):
    cmd_j =  cmd_prefix + ' ' + fname_config + ' ' + fname_data + ' ' + fname_nmatrix + ' ' + str(ii_gen) + ' ' + str(j)
    jobname = 'paraProp-job-' + str(j)
    sh_file = jobname + '.sh'
    out_file = jobname + '.out'
    make_job(sh_file, out_file, jobname, cmd_j)
    submit_job(sh_file)

proceed_bool = False
while proceed_bool == False:
    tsleep = 10.
    nJobs = countjobs()
    if nJobs > 0:
        time.sleep(tsleep)
    else:
        proceed_bool = True

ii_gen += 1
#Wait for jobs to be submitted

while ii_gen < GA_1.nGenerations:
    nJobs = countjobs()
    tsleep = 30.
    print('Check jobs')
    if nJobs == 0:
        print('Submitted jobs')
        # APPLY GA selection
        nmatrix_hdf = h5py.File(fname_nmatrix, 'r+')
        S_arr = np.array(nmatrix_hdf['S_arr'])
        n_profile_matrix = np.array(nmatrix_hdf['n_profile_matrix'])
        n_profile_initial = n_profile_matrix[0]
        n_profile_parents = n_profile_matrix[ii_gen - 1]
        S_list = S_arr[ii_gen - 1]
        print(ii_gen - 1)

        # n_profile_children = roulette(n_profile_parents, S_list, n_profile_initial)
        n_profile_children = selection(prof_list=n_profile_parents, S_list=S_list, prof_list_initial=n_prof_pool)
        n_profile_matrix[ii_gen] = n_profile_children
        nmatrix_hdf.close()
        for j in range(GA_1.nIndividuals):
            #Create Command
            dir_outfiles0 = 'outfiles'
            dir_shfiles0 = 'shfiles'

            dir_outfiles = dir_outfiles0 + '/' + 'gen' + str(ii_gen)
            dir_shfiles = dir_shfiles0 + '/' + 'gen' + str(ii_gen)
            if os.path.isdir(dir_outfiles) == False:
                os.system('mkdir ' + dir_outfiles)
            if os.path.isdir(dir_outfiles) == False:
                os.system('mkdir ' + dir_shfiles)

            cmd_j = cmd_prefix + ' ' + fname_config + ' ' + fname_data + ' ' + fname_nmatrix + ' ' + str(ii_gen) + ' ' + str(j)
            jobname = 'paraProp-job-' + ii_gen + '-' + str(j)
            sh_file = jobname + '.sh'
            out_file = dir_outfiles + '/' + jobname + '.out'
            make_job(sh_file, out_file, jobname, cmd_j)
            submit_job(sh_file)
            #After Jobs are submitted
            os.system('mv ' + sh_file + ' ' + dir_outfiles)
        ii_gen += 1
        print('Jobs running -> Generation: ', ii_gen)
    else:
        print('Queue of jobs: ', nJobs)
        print('Wait:', tsleep, ' seconds')
        time.sleep(tsleep)