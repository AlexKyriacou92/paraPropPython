import datetime
import os
import random

import h5py
import numpy as np
from scipy.interpolate import interp1d

from makeSim import createMatrix
from genetic_functions import initialize_from_analytical, roulette, initalize_from_fluctuations
from pleiades_scripting import make_command, test_job, submit_job, test_job_data, make_command_data, make_job
from selection_functions import selection

import subprocess
import time

def save_profile(fname_profile, z_profile, n_profile):
    nDepths = len(z_profile)
    fout = open(fname_profile, 'w')
    for i in range(nDepths):
        z_i = z_profile[i]
        n_i = n_profile[i]
        line = str(round(z_i, 3)) + '\t' + str(round(n_i, 3)) + '\n'
        fout.write(line)
    fout.close()
    return -1

nStart = 10000
#To start with -> just run 15 individuals
nIndividuals = 100
nGens = 10

fname_start = 'start_profiles/aletsch_glacier_2.txt'
#fname_test = 'start_profiles/parallel-profile-0605_2nd-pk.txt'

data_nprof0 = np.genfromtxt(fname_start)
zprof_0 = data_nprof0[:,0]
nprof_0 = data_nprof0[:,1]
nDepths0 = len(zprof_0)

#data_test = np.genfromtxt(fname_test)
#n_prof_test = data_test[:,1]


f_prof_interp = interp1d(zprof_0, nprof_0)

dz_start = zprof_0[1] - zprof_0[0]
dz_small = 0.5

if dz_small != dz_start:
    factor = int(dz_start/dz_small)

    zprof_1 = np.arange(min(zprof_0), max(zprof_0) + dz_small, dz_small)
    nDepths = len(zprof_1)

    nprof_1 = np.ones(nDepths)
    nprof_1[0] = nprof_0[0]
    nprof_1[-1] = nprof_0[-1]
    nprof_1[1:-1] = f_prof_interp(zprof_1[1:-1])
else:
    zprof_1 = zprof_0
    nprof_1 = nprof_0
    nDepths = len(zprof_1)


#Initialize Profiles

nHalf = nStart//4

n_prof_pool = initialize_from_analytical(nprof_1, 0.04*np.ones(len(nprof_1)), nHalf)
n_prof_pool2 = initalize_from_fluctuations(nprof_1, zprof_1, nHalf)

for i in range(nHalf):
    n_prof_pool.append(n_prof_pool2[i])

for i in range(nHalf):
    n_const = 0.8 * random.random()
    n_prof_flat = np.ones(nDepths) + n_const
    n_prof_pool.append(n_prof_flat)
from math import pi
for i in range(nHalf):
    amp_rand = 0.4*random.random()
    z_period = random.uniform(0.5,15)
    k_factor = 1/z_period
    phase_rand = random.uniform(0, 2*pi)
    freq_rand = amp_rand*np.sin(2*pi*zprof_0*k_factor + phase_rand)
    n_prof_flat = np.ones(nDepths) + n_const
    n_prof_pool.append(n_prof_flat)
random.shuffle(n_prof_pool)
n_prof_initial = n_prof_pool[:nIndividuals]

S_arr = np.zeros((nGens, nIndividuals))
n_prof_array = np.ones((nGens, nIndividuals, nDepths))


n_prof_psuedo_data = nprof_1
n_min = 1.1
n_max = 1.8

for i in range(nDepths):
    if n_prof_psuedo_data[i] < n_min:
        n_prof_psuedo_data[i] = n_min
    elif n_prof_psuedo_data[i] > n_max:
        n_prof_psuedo_data[i] = n_max

fname_nprof_pseudo = 'test_nprof_start.txt'
save_profile(fname_nprof_pseudo, zprof_1, nprof_1)

#=======================================================================================
#Create n-matrix
fname_config = 'config_aletsch.txt'
fname_nmatrix = 'test_nmatrix_data.h5'

createMatrix(fname_config=fname_config, n_prof_initial=n_prof_initial, z_profile=zprof_1,
             fname_nmatrix=fname_nmatrix, nGenerations = nGens)

#Create Pseudo_Data

fname_output_pseudo = 'psuedo_data.h5'
os.system('python runSim_pseudo_data.py ' + fname_config + ' ' + fname_nprof_pseudo + ' ' + fname_output_pseudo)

#=========================================================================================

def countjobs():
    cmd = 'squeue | grep "kyriacou" | wc -l'
    try:
        output = int(subprocess.check_output(cmd, shell=True))
    except:
        output = 0
    return output

def make_command_pseudodata(config_file, bscan_data_file, nprof_matrix_file, ii, jj):
    command = 'python runSim_nprofile_psuedoFT_test.py ' + config_file + ' ' + bscan_data_file + ' ' + nprof_matrix_file + ' ' + str(ii) + ' ' + str(jj)
    return command

def test_job_pseudodata(prefix, config_file, bscan_data_file, nprof_matrix_file, gene, individual):
    nprof_h5 = h5py.File(nprof_matrix_file, 'r')
    nprof_matrix = np.array(nprof_h5.get('n_profile_matrix'))
    nprof_list = nprof_matrix[gene]
    nProf = len(nprof_list)
    nprof_h5.close()

    fname_joblist = prefix + '-joblist.txt'
    fout_joblist = open(fname_joblist, 'w')
    fout_joblist.write('Joblist ' + prefix + '\n')
    fout_joblist.write(
        prefix + '\t' + config_file + '\t' + bscan_data_file + '\t' + nprof_matrix_file + '\t' + str(gene) + '\n')
    fout_joblist.write('shell_file' + '\t' + 'output_file' + '\t' + 'prof_number' + '\n \n')
    command = make_command_pseudodata(config_file, bscan_data_file, nprof_matrix_file, gene, individual)
    jobname = 'job-' + str(individual)
    fname_shell = prefix + jobname + '.sh'
    fname_out = prefix + jobname + '.out'
    line = fname_shell + '\t' + fname_out + '\t' + str(individual) + '\n'
    fout_joblist.write(line)
    make_job(fname_shell, fname_out, jobname, command)
    return fname_shell

jj = 1
nMinutes = 1
minutes_s = 60.0
t_sleep = nMinutes * minutes_s
while jj + 1 < nGens:
    nJobs = countjobs()
    if nJobs == 0:
        nmatrix_hdf = h5py.File(fname_nmatrix, 'r+')
        S_arr = nmatrix_hdf['S_arr']
        n_profile_matrix = nmatrix_hdf['n_profile_matrix']

        n_profile_initial = n_profile_matrix[0]
        n_profile_parents = n_profile_matrix[jj - 1]
        S_list = S_arr[jj - 1]
        print(jj - 1)

        #n_profile_children = roulette(n_profile_parents, S_list, n_profile_initial)
        n_profile_children = selection(prof_list=n_profile_parents, S_list=S_list, prof_list_initial=n_prof_pool)
        n_profile_matrix[jj] = n_profile_children
        nmatrix_hdf.close()

        nIndividuals = len(n_profile_children)
        for ii in range(nIndividuals):
            fname_shell = test_job_pseudodata(prefix='test', config_file=fname_config, bscan_data_file=fname_output_pseudo,
                                   nprof_matrix_file=fname_nmatrix, gene=jj, individual=ii)
            submit_job(fname_shell)
        jj += 1
    else:
        print('generation: ', jj-1, ',', nJobs, 'remaining, wait ', t_sleep, ' seconds')
        print(datetime.datetime.now())
        time.sleep(t_sleep)

fname_list = [fname_config, fname_start, fname_nprof_pseudo, fname_output_pseudo, fname_nmatrix]
destination_dir = '/common/home/akyriacou/paraPropPython/paraPropPython_multisim/genetic_algorithm_analysis/paraPropData/GA_test/'
account = 'akyriacou@astro22.physik.uni-wuppertal.de'
cmd = 'scp '
for i in range(len(fname_list)):
    cmd += fname_list[i] + ' '
cmd += account + ':' + destination_dir
os.system(cmd)