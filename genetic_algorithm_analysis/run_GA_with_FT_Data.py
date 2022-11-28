import datetime
import os
import random

import h5py
import numpy as np
from scipy.interpolate import interp1d

from makeSim import createMatrix
from genetic_functions import initialize_from_analytical, roulette, initalize_from_fluctuations
from pleiades_scripting import make_command, test_job, submit_job, test_job_data
import subprocess
import time

nStart = 10000
#To start with -> just run 15 individuals
nIndividuals = 15
nGens = 90

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

#=======================================================================================
#Create n-matrix
fname_config = 'config_aletsch.txt'
fname_nmatrix = 'test_nmatrix_data.h5'
fname_data = 'Field-Test-data.h5'
createMatrix(fname_config=fname_config, n_prof_initial=n_prof_initial, z_profile=zprof_0,
             fname_nmatrix=fname_nmatrix, nGenerations = nGens)
#=========================================================================================

def countjobs():
    cmd = 'squeue | grep "kyriacou" | wc -l'
    try:
        output = int(subprocess.check_output(cmd, shell=True))
    except:
        output = 0
    return output

jj = 1
nMinutes = 1
minutes_s = 60.0
t_sleep = nMinutes * minutes_s
while jj < nGens:
    nJobs = countjobs()
    if nJobs == 0:
        nmatrix_hdf = h5py.File(fname_nmatrix, 'r+')
        S_arr = nmatrix_hdf['S_arr']
        n_profile_matrix = nmatrix_hdf['n_profile_matrix']

        n_profile_initial = n_profile_matrix[0]
        n_profile_parents = n_profile_matrix[jj - 1]
        S_list = S_arr[jj - 1]
        print(jj - 1)

        n_profile_children = roulette(n_profile_parents, S_list, n_profile_initial)
        n_profile_matrix[jj] = n_profile_children
        nmatrix_hdf.close()

        nIndividuals = len(n_profile_children)
        for ii in range(nIndividuals):
            fname_shell = test_job_data(prefix='test', config_file=fname_config, bscan_data_file=fname_data,
                                   nprof_matrix_file=fname_nmatrix, gene=jj, individual=ii)
            submit_job(fname_shell)
        jj += 1
    else:
        print('generation: ', jj-1, ',', nJobs, 'remaining, wait ', t_sleep, ' seconds')
        print(datetime.datetime.now())
        time.sleep(t_sleep)


fname_list = [fname_config, fname_start, fname_data, fname_data, fname_nmatrix]
destination_dir = '/common/home/akyriacou/paraPropPython/paraPropPython_multisim/genetic_algorithm_analysis/paraPropData/GA_test/'
account = 'akyriacou@astro22.physik.uni-wuppertal.de'
cmd = 'scp '
for i in range(len(fname_list)):
    cmd += fname_list[i] + ' '
cmd += account + ':' + destination_dir
os.system(cmd)